# 第二章：Agent系统详解

## 2.1 Agent 类层次结构

```
BaseAgent (基类)
├── ReActAgent (ReAct 模式基类)
│   └── ToolCallAgent (工具调用 Agent)
│       ├── Manus (通用主 Agent)
│       ├── MCPAgent (MCP 协议 Agent)
│       ├── BrowserAgent (浏览器控制 Agent)
│       └── DataAnalysis (数据分析 Agent)
```

## 2.2 BaseAgent 核心实现

**文件位置**：`app/agent/base.py`

### 2.2.1 核心属性

```python
class BaseAgent(BaseModel, ABC):
    # 核心属性
    name: str                           # Agent 唯一名称
    description: Optional[str]          # Agent 描述

    # 提示词
    system_prompt: Optional[str]       # 系统级指令
    next_step_prompt: Optional[str]     # 下一步提示

    # 依赖组件
    llm: LLM                            # 语言模型实例
    memory: Memory                      # 记忆存储
    state: AgentState                   # 当前状态

    # 执行控制
    max_steps: int = 10                 # 最大步数
    current_step: int = 0               # 当前步数
    duplicate_threshold: int = 2         # 重复检测阈值
```

### 2.2.2 状态管理

```python
class AgentState(str, Enum):
    IDLE = "IDLE"       # 空闲状态
    RUNNING = "RUNNING" # 运行中
    FINISHED = "FINISHED"  # 已完成
    ERROR = "ERROR"      # 错误状态
```

**状态转换上下文管理器**：
```python
@asynccontextmanager
async def state_context(self, new_state: AgentState):
    """安全的 agent 状态转换"""
    previous_state = self.state
    self.state = new_state
    try:
        yield
    except Exception as e:
        self.state = AgentState.ERROR
        raise e
    finally:
        self.state = previous_state
```

### 2.2.3 执行循环

```python
async def run(self, request: Optional[str] = None) -> str:
    """Agent 主执行循环"""
    if self.state != AgentState.IDLE:
        raise RuntimeError(f"Cannot run agent from state: {self.state}")

    if request:
        self.update_memory("user", request)

    results: List[str] = []
    async with self.state_context(AgentState.RUNNING):
        while (
            self.current_step < self.max_steps
            and self.state != AgentState.FINISHED
        ):
            self.current_step += 1
            step_result = await self.step()

            # 检测卡死状态
            if self.is_stuck():
                self.handle_stuck_state()

            results.append(f"Step {self.current_step}: {step_result}")

    await SANDBOX_CLIENT.cleanup()
    return "\n".join(results)
```

### 2.2.4 内存管理

```python
def update_memory(
    self,
    role: ROLE_TYPE,
    content: str,
    base64_image: Optional[str] = None,
    **kwargs,
) -> None:
    """添加消息到 Agent 记忆"""
    message_map = {
        "user": Message.user_message,
        "system": Message.system_message,
        "assistant": Message.assistant_message,
        "tool": lambda content, **kw: Message.tool_message(content, **kw),
    }
    self.memory.add_message(message_map[role](content, **kwargs))
```

### 2.2.5 卡死检测

```python
def is_stuck(self) -> bool:
    """通过检测重复内容判断 Agent 是否卡死"""
    if len(self.memory.messages) < 2:
        return False

    last_message = self.memory.messages[-1]
    duplicate_count = sum(
        1
        for msg in reversed(self.memory.messages[:-1])
        if msg.role == "assistant" and msg.content == last_message.content
    )
    return duplicate_count >= self.duplicate_threshold
```

## 2.3 ReActAgent 实现

**文件位置**：`app/agent/react.py`

ReAct (Reasoning + Acting) 模式将推理和执行分离：

```python
class ReActAgent(BaseAgent):
    """ReAct 模式基类"""

    async def step(self) -> str:
        """执行单步操作"""
        # 1. Think - 思考下一步
        think_result = await self.think()
        if not think_result:
            return "思考完成，无更多行动"

        # 2. Act - 执行工具调用
        act_result = await self.act()
        return act_result

    @abstractmethod
    async def think(self) -> bool:
        """思考阶段 - 决定下一步动作"""
        pass

    @abstractmethod
    async def act(self) -> str:
        """执行阶段 - 执行工具调用"""
        pass
```

## 2.4 ToolCallAgent 实现

**文件位置**：`app/agent/toolcall.py`

### 2.4.1 think() 方法

```python
async def think(self) -> bool:
    """通过工具决定下一步行动"""
    if self.next_step_prompt:
        user_msg = Message.user_message(self.next_step_prompt)
        self.messages += [user_msg]

    # 调用 LLM 获取响应
    response = await self.llm.ask_tool(
        messages=self.messages,
        system_msgs=[Message.system_message(self.system_prompt)] if self.system_prompt else None,
        tools=self.available_tools.to_params(),
        tool_choice=self.tool_choices,
    )

    self.tool_calls = response.tool_calls if response else []
    content = response.content if response else ""

    # 添加助手消息到记忆
    assistant_msg = Message.from_tool_calls(
        content=content,
        tool_calls=self.tool_calls
    ) if self.tool_calls else Message.assistant_message(content)
    self.memory.add_message(assistant_msg)

    return bool(self.tool_calls)
```

### 2.4.2 act() 方法

```python
async def act(self) -> str:
    """执行工具调用"""
    if not self.tool_calls:
        return self.messages[-1].content or "No content"

    results = []
    for command in self.tool_calls:
        result = await self.execute_tool(command)
        results.append(result)

        # 添加工具响应到记忆
        tool_msg = Message.tool_message(
            content=result,
            tool_call_id=command.id,
            name=command.function.name,
        )
        self.memory.add_message(tool_msg)

    return "\n\n".join(results)
```

### 2.4.3 工具执行

```python
async def execute_tool(self, command: ToolCall) -> str:
    """执行单个工具调用"""
    name = command.function.name
    if name not in self.available_tools.tool_map:
        return f"Error: Unknown tool '{name}'"

    try:
        args = json.loads(command.function.arguments or "{}")
        result = await self.available_tools.execute(name=name, tool_input=args)

        # 处理特殊工具
        await self._handle_special_tool(name=name, result=result)

        observation = f"Observed output of cmd `{name}` executed:\n{str(result)}"
        return observation
    except json.JSONDecodeError:
        return f"Error parsing arguments for {name}: Invalid JSON"
    except Exception as e:
        return f"Error: {e}"
```

## 2.5 Manus Agent 实现

**文件位置**：`app/agent/manus.py`

### 2.5.1 类定义

```python
class Manus(ToolCallAgent):
    """通用的多工具 Agent，支持本地和 MCP 工具"""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks"

    max_steps: int = 20
    max_observe: int = 10000

    # MCP 客户端
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 可用工具集合
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),    # Python 执行
            BrowserUseTool(),   # 浏览器控制
            StrReplaceEditor(), # 文件编辑
            AskHuman(),         # 人工确认
            Terminate(),        # 终止执行
        )
    )

    special_tool_names: list[str] = Field(
        default_factory=lambda: [Terminate().name]
    )
```

### 2.5.2 MCP 服务器初始化

```python
@classmethod
async def create(cls, **kwargs) -> "Manus":
    """工厂方法 - 创建并初始化 Manus 实例"""
    instance = cls(**kwargs)
    await instance.initialize_mcp_servers()
    instance._initialized = True
    return instance

async def initialize_mcp_servers(self) -> None:
    """初始化连接的 MCP 服务器"""
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

### 2.5.3 think() 扩展

```python
async def think(self) -> bool:
    """处理当前状态并决定下一步行动"""
    if not self._initialized:
        await self.initialize_mcp_servers()
        self._initialized = True

    # 检查是否使用浏览器
    browser_in_use = any(
        tc.function.name == BrowserUseTool().name
        for msg in self.memory.messages[-3:]
        if msg.tool_calls
        for tc in msg.tool_calls
    )

    # 如果使用浏览器，添加浏览器上下文
    if browser_in_use:
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )

    return await super().think()
```

## 2.6 BrowserAgent 实现

**文件位置**：`app/agent/browser.py`

```python
class BrowserAgent(ToolCallAgent):
    """专门用于浏览器控制的 Agent"""

    name: str = "browser"
    max_steps: int = 20

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            BrowserUseTool(),
            Terminate()
        )
    )

    async def think(self) -> bool:
        """获取浏览器状态并决定下一步"""
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )
        return await super().think()
```

### BrowserContextHelper

```python
class BrowserContextHelper:
    """浏览器上下文辅助类"""

    async def format_next_step_prompt(self) -> str:
        """获取浏览器状态并格式化提示词"""
        browser_state = await self.get_browser_state()

        if browser_state:
            url_info = f"\n   URL: {browser_state.get('url')}"
            tabs_info = f"\n   {len(browser_state.get('tabs', []))} tab(s)"
            # ... 其他状态信息

            # 如果有截图，添加到消息
            if self._current_base64_image:
                self.agent.memory.add_message(
                    Message.user_message(
                        content="Current browser screenshot:",
                        base64_image=self._current_base64_image
                    )
                )

        return NEXT_STEP_PROMPT.format(...)
```

## 2.7 执行流程图

```
用户输入
    │
    ▼
┌─────────────────┐
│  BaseAgent.run()│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     状态检查 (IDLE?)             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     状态设置为 RUNNING           │
└────────┬────────────────────────┘
         │
         ▼
    ┌────┴────┐
    │ 循环开始 │
    └────┬────┘
         │
         ▼
┌─────────────────────────────────┐
│  ToolCallAgent.think()         │
│  - 添加 next_step_prompt        │
│  - 调用 LLM.ask_tool()          │
│  - 解析工具调用                 │
│  - 更新记忆                     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ToolCallAgent.act()           │
│  - 遍历 tool_calls             │
│  - 执行 execute_tool()         │
│  - 处理工具结果                 │
│  - 更新记忆                     │
└────────┬────────────────────────┘
         │
         ▼
    ┌────┴────┐
    │ 状态==  │
    │FINISHED?│
    └────┬────┘
         │
    ┌────┴────┐
    │  是    │────► 结束
    └────────┘
         │
         ▼
    ┌────────┐
    │  否    │
    └────┬───┘
         │
         ▼
    ┌────┴────┐
    │步数<max?│
    └────┬────┘
         │
    ┌────┴────┐
    │  是    │────► 继续循环
    └────────┘
         │
         ▼
    ┌────────┐
    │  否    │────► 达到最大步数，结束
    └────────┘
```
