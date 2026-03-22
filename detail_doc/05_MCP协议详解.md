# 第五章：MCP协议详解

## 5.1 MCP 协议概述

MCP (Model Context Protocol) 是一种标准化协议，允许 AI 应用与外部工具和数据源连接。

```
┌─────────────────┐                    ┌─────────────────┐
│                 │   MCP Protocol      │                 │
│   OpenManus     │◄──────────────────►│   MCP Server    │
│   (MCP Client)  │                    │   (如 Filesystem │
│                 │                    │    Server)      │
└─────────────────┘                    └─────────────────┘
        │                                       │
        │  连接并发现工具                        │
        │  ─────────────                         │
        │  1. connect_sse() / connect_stdio()   │
        │  2. list_tools()                      │
        │  3. call_tool()                        │
        │                                       │
        ▼                                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    可用工具集合                              │
│  - 本地工具 (Python, Bash, Editor)                          │
│  - MCP 远程工具 (mcp_server1_tool1, mcp_server1_tool2)      │
└─────────────────────────────────────────────────────────────┘
```

## 5.2 MCP 客户端实现

**文件位置**：`app/tool/mcp.py`

### 5.2.1 MCPClients 类

```python
class MCPClients(ToolCollection):
    """MCP 客户端管理器"""

    sessions: Dict[str, ClientSession] = {}
    exit_stacks: Dict[str, AsyncExitStack] = {}

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """通过 SSE 传输连接 MCP 服务器"""
        server_id = server_id or server_url

        # 确保断开旧的连接
        if server_id in self.sessions:
            await self.disconnect(server_id)

        # 创建异步上下文
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        # 建立 SSE 连接
        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        # 初始化并列出工具
        await self._initialize_and_list_tools(server_id)
```

### 5.2.2 Stdio 传输连接

```python
async def connect_stdio(
    self,
    command: str,
    args: List[str],
    server_id: str = ""
) -> None:
    """通过 stdio 传输连接 MCP 服务器"""
    server_id = server_id or command

    if server_id in self.sessions:
        await self.disconnect(server_id)

    exit_stack = AsyncExitStack()
    self.exit_stacks[server_id] = exit_stack

    # 配置 stdio 参数
    server_params = StdioServerParameters(
        command=command,
        args=args
    )

    # 建立 stdio 连接
    stdio_transport = await exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    read, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    self.sessions[server_id] = session

    await self._initialize_and_list_tools(server_id)
```

### 5.2.3 工具发现与注册

```python
async def _initialize_and_list_tools(self, server_id: str) -> None:
    """初始化会话并填充工具映射"""
    session = self.sessions.get(server_id)
    if not session:
        raise RuntimeError(f"Session not initialized for server {server_id}")

    # 初始化会话
    await session.initialize()

    # 列出可用工具
    response = await session.list_tools()

    # 为每个服务器工具创建代理
    for tool in response.tools:
        original_name = tool.name

        # 工具名称格式化：mcp_{server_id}_{tool_name}
        tool_name = f"mcp_{server_id}_{original_name}"
        tool_name = self._sanitize_tool_name(tool_name)

        # 创建 MCP 客户端工具代理
        server_tool = MCPClientTool(
            name=tool_name,
            description=tool.description,
            parameters=tool.inputSchema,
            session=session,
            server_id=server_id,
            original_name=original_name,
        )
        self.tool_map[tool_name] = server_tool

    # 更新工具元组
    self.tools = tuple(self.tool_map.values())

def _sanitize_tool_name(self, name: str) -> str:
    """清理工具名称"""
    import re
    # 替换无效字符
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # 移除连续下划线
    sanitized = re.sub(r"_+", "_", sanitized)
    # 去除首尾下划线
    sanitized = sanitized.strip("_")
    # 截断到 64 字符
    if len(sanitized) > 64:
        sanitized = sanitized[:64]
    return sanitized
```

### 5.2.4 断开连接

```python
async def disconnect(self, server_id: str = "") -> None:
    """断开与 MCP 服务器的连接"""
    if server_id:
        # 断开特定服务器
        if server_id in self.sessions:
            try:
                exit_stack = self.exit_stacks.get(server_id)
                if exit_stack:
                    await exit_stack.aclose()

                # 清理引用
                self.sessions.pop(server_id, None)
                self.exit_stacks.pop(server_id, None)

                # 移除关联的工具
                self.tool_map = {
                    k: v for k, v in self.tool_map.items()
                    if v.server_id != server_id
                }
                self.tools = tuple(self.tool_map.values())
            except Exception as e:
                logger.error(f"Error disconnecting from server {server_id}: {e}")
    else:
        # 断开所有服务器
        for sid in sorted(list(self.sessions.keys())):
            await self.disconnect(sid)
```

## 5.3 MCPClientTool 工具代理

```python
class MCPClientTool(BaseTool):
    """MCP 服务器工具代理"""

    session: Optional[ClientSession] = None
    server_id: str = ""
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        """通过 MCP 服务器执行工具"""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            logger.info(f"Executing tool: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)

            # 解析返回内容
            content_str = ", ".join(
                item.text for item in result.content
                if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")
```

## 5.4 MCP Server 实现

**文件位置**：`app/mcp/server.py`

### 5.4.1 MCPServer 类

```python
class MCPServer:
    """MCP 服务器实现"""

    def __init__(self, name: str = "openmanus"):
        self.server = FastMCP(name)
        self.tools: Dict[str, BaseTool] = {}

        # 初始化标准工具
        self.tools["bash"] = Bash()
        self.tools["browser"] = BrowserUseTool()
        self.tools["editor"] = StrReplaceEditor()
        self.tools["terminate"] = Terminate()
```

### 5.4.2 工具注册

```python
def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
    """注册工具到 MCP 服务器"""
    tool_name = method_name or tool.name
    tool_param = tool.to_param()
    tool_function = tool_param["function"]

    # 定义异步工具方法
    async def tool_method(**kwargs):
        logger.info(f"Executing {tool_name}: {kwargs}")
        result = await tool.execute(**kwargs)
        logger.info(f"Result of {tool_name}: {result}")

        # 处理结果格式
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump())
        elif isinstance(result, dict):
            return json.dumps(result)
        return result

    # 设置元数据
    tool_method.__name__ = tool_name
    tool_method.__doc__ = self._build_docstring(tool_function)
    tool_method.__signature__ = self._build_signature(tool_function)

    # 注册参数模式
    param_props = tool_function.get("parameters", {}).get("properties", {})
    required_params = tool_function.get("parameters", {}).get("required", [])
    tool_method._parameter_schema = {
        param_name: {
            "description": param_details.get("description", ""),
            "type": param_details.get("type", "any"),
            "required": param_name in required_params,
        }
        for param_name, param_details in param_props.items()
    }

    # 注册到服务器
    self.server.tool()(tool_method)
    logger.info(f"Registered tool: {tool_name}")
```

### 5.4.3 签名构建

```python
def _build_signature(self, tool_function: dict) -> Signature:
    """从工具函数元数据构建签名"""
    param_props = tool_function.get("parameters", {}).get("properties", {})
    required_params = tool_function.get("parameters", {}).get("required", [])

    parameters = []

    for param_name, param_details in param_props.items():
        param_type = param_details.get("type", "")
        default = Parameter.empty if param_name in required_params else None

        # JSON Schema 类型映射到 Python 类型
        annotation = Any
        if param_type == "string":
            annotation = str
        elif param_type == "integer":
            annotation = int
        elif param_type == "number":
            annotation = float
        elif param_type == "boolean":
            annotation = bool
        elif param_type == "object":
            annotation = dict
        elif param_type == "array":
            annotation = list

        param = Parameter(
            name=param_name,
            kind=Parameter.KEYWORD_ONLY,
            default=default,
            annotation=annotation,
        )
        parameters.append(param)

    return Signature(parameters=parameters)
```

### 5.4.4 运行服务器

```python
def register_all_tools(self) -> None:
    """注册所有工具"""
    for tool in self.tools.values():
        self.register_tool(tool)

def run(self, transport: str = "stdio") -> None:
    """运行 MCP 服务器"""
    self.register_all_tools()
    atexit.register(lambda: asyncio.run(self.cleanup()))

    logger.info(f"Starting OpenManus server ({transport} mode)")
    self.server.run(transport=transport)
```

## 5.5 MCP 配置

**配置文件**：`config/mcp.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    },
    "github": {
      "type": "sse",
      "url": "http://localhost:3000/sse"
    }
  }
}
```

## 5.6 使用流程

```
1. Agent 初始化
   └── Manus.create()
       └── initialize_mcp_servers()
           └── 读取 config.mcp_config.servers

2. 连接每个 MCP 服务器
   ├── SSE 连接: mcp_clients.connect_sse(url, server_id)
   └── Stdio 连接: mcp_clients.connect_stdio(command, args, server_id)

3. 工具发现
   └── _initialize_and_list_tools()
       └── session.list_tools()
           └── 为每个工具创建 MCPClientTool

4. 添加工具到 Agent
   └── available_tools.add_tools(*new_tools)

5. Agent 执行时使用 MCP 工具
   └── execute_tool(command)
       └── mcp_client_tool.execute(**kwargs)
           └── session.call_tool(original_name, kwargs)
```

## 5.7 与 Manus Agent 集成

```python
class Manus(ToolCallAgent):
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        for server_id, server_config in config.mcp_config.servers.items():
            if server_config.type == "sse":
                await self.connect_mcp_server(server_config.url, server_id)
            elif server_config.type == "stdio":
                await self.connect_mcp_server(
                    server_config.command, server_id,
                    use_stdio=True,
                    stdio_args=server_config.args
                )
```
