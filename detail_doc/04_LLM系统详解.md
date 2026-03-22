# 第四章：LLM系统详解

## 4.1 LLM 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  _instances: Dict[str, LLM]  (单例缓存)                │ │
│  │  client: AsyncOpenAI / AsyncAzureOpenAI / BedrockClient │
│  │  tokenizer: tiktoken                                   │ │
│  │  token_counter: TokenCounter                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │   ask()   │    │ask_tool() │    │ask_with_   │        │
│  │  (基础对话) │    │(工具调用)  │    │ images()   │        │
│  └────────────┘    └────────────┘    └────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 核心类实现

**文件位置**：`app/llm.py`

### 4.2.1 单例模式实现

```python
class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", llm_config=None):
        # 单例缓存
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]
```

### 4.2.2 多后端支持

```python
def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
    llm_config = llm_config or config.llm
    llm_config = llm_config.get(config_name, llm_config["default"])

    self.model = llm_config.model
    self.max_tokens = llm_config.max_tokens
    self.temperature = llm_config.temperature
    self.api_type = llm_config.api_type
    self.api_key = llm_config.api_key
    self.base_url = llm_config.base_url

    # 根据 api_type 选择客户端
    if self.api_type == "azure":
        self.client = AsyncAzureOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            api_version=self.api_version,
        )
    elif self.api_type == "aws":
        self.client = BedrockClient()  # AWS Bedrock
    else:
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
```

## 4.3 Token 计数系统

### 4.3.1 TokenCounter 类

```python
class TokenCounter:
    """Token 计数工具"""

    # Token 常量
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # 图像处理常量
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的 token 总数"""
        total_tokens = self.FORMAT_TOKENS

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS
            tokens += self.count_text(message.get("role", ""))

            if "content" in message:
                tokens += self.count_content(message["content"])

            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            total_tokens += tokens

        return total_tokens

    def count_image(self, image_item: dict) -> int:
        """计算图像的 token 数量"""
        detail = image_item.get("detail", "medium")

        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        if detail == "high" or detail == "medium":
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return 1024  # 默认值
```

### 4.3.2 Token 限制检查

```python
def check_token_limit(self, input_tokens: int) -> bool:
    """检查是否超过 token 限制"""
    if self.max_input_tokens is not None:
        return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
    return True

def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
    """更新 token 计数"""
    self.total_input_tokens += input_tokens
    self.total_completion_tokens += completion_tokens
```

## 4.4 消息格式化

### 4.4.1 format_messages 方法

```python
@staticmethod
def format_messages(
    messages: List[Union[dict, Message]],
    supports_images: bool = False
) -> List[dict]:
    """将 Message 对象转换为 LLM 格式"""
    formatted_messages = []

    for message in messages:
        # 转换 Message 对象
        if isinstance(message, Message):
            message = message.to_dict()

        if isinstance(message, dict):
            # 处理 base64 图像
            if supports_images and message.get("base64_image"):
                if not message.get("content"):
                    message["content"] = []
                elif isinstance(message["content"], str):
                    message["content"] = [
                        {"type": "text", "text": message["content"]}
                    ]

                message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{message['base64_image']}"
                    }
                })
                del message["base64_image"]

            if "content" in message or "tool_calls" in message:
                formatted_messages.append(message)

    return formatted_messages
```

## 4.5 ask() 方法 - 基础对话

```python
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((OpenAIError, Exception, ValueError)),
)
async def ask(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    stream: bool = True,
    temperature: Optional[float] = None,
) -> str:
    """发送对话请求"""
    supports_images = self.model in MULTIMODAL_MODELS

    # 格式化消息
    if system_msgs:
        system_msgs = self.format_messages(system_msgs, supports_images)
        messages = system_msgs + self.format_messages(messages, supports_images)
    else:
        messages = self.format_messages(messages, supports_images)

    # Token 检查
    input_tokens = self.count_message_tokens(messages)
    if not self.check_token_limit(input_tokens):
        raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

    params = {
        "model": self.model,
        "messages": messages,
    }

    if self.model in REASONING_MODELS:
        params["max_completion_tokens"] = self.max_tokens
    else:
        params["max_tokens"] = self.max_tokens
        params["temperature"] = temperature if temperature is not None else self.temperature

    if not stream:
        # 非流式请求
        response = await self.client.chat.completions.create(**params, stream=False)
        self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    # 流式请求
    self.update_token_count(input_tokens)
    response = await self.client.chat.completions.create(**params, stream=True)

    collected_messages = []
    async for chunk in response:
        chunk_message = chunk.choices[0].delta.content or ""
        collected_messages.append(chunk_message)
        print(chunk_message, end="", flush=True)

    return "".join(collected_messages).strip()
```

## 4.6 ask_tool() 方法 - 工具调用

```python
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
    temperature: Optional[float] = None,
    **kwargs,
) -> ChatCompletionMessage | None:
    """发送工具调用请求"""
    # 验证 tool_choice
    if tool_choice not in TOOL_CHOICE_VALUES:
        raise ValueError(f"Invalid tool_choice: {tool_choice}")

    supports_images = self.model in MULTIMODAL_MODELS

    # 格式化消息
    if system_msgs:
        system_msgs = self.format_messages(system_msgs, supports_images)
        messages = system_msgs + self.format_messages(messages, supports_images)
    else:
        messages = self.format_messages(messages, supports_images)

    # Token 计算
    input_tokens = self.count_message_tokens(messages)
    if tools:
        for tool in tools:
            input_tokens += self.count_tokens(str(tool))

    # Token 限制检查
    if not self.check_token_limit(input_tokens):
        raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

    # 设置请求参数
    params = {
        "model": self.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        **kwargs,
    }

    if self.model in REASONING_MODELS:
        params["max_completion_tokens"] = self.max_tokens
    else:
        params["max_tokens"] = self.max_tokens
        params["temperature"] = temperature if temperature is not None else self.temperature

    params["stream"] = False
    response = await self.client.chat.completions.create(**params)

    if not response.choices or not response.choices[0].message:
        return None

    self.update_token_count(
        response.usage.prompt_tokens,
        response.usage.completion_tokens
    )

    return response.choices[0].message
```

## 4.7 ask_with_images() - 多模态支持

```python
async def ask_with_images(
    self,
    messages: List[Union[dict, Message]],
    images: List[Union[str, dict]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    stream: bool = False,
) -> str:
    """发送带图像的多模态请求"""
    if self.model not in MULTIMODAL_MODELS:
        raise ValueError(f"Model {self.model} does not support images")

    formatted_messages = self.format_messages(messages, supports_images=True)

    # 确保最后一条消息是用户消息
    last_message = formatted_messages[-1]

    # 添加图像到消息
    multimodal_content = [{"type": "text", "text": last_message.get("content", "")}]

    for image in images:
        multimodal_content.append({
            "type": "image_url",
            "image_url": {"url": image} if isinstance(image, str) else image
        })

    last_message["content"] = multimodal_content

    # ... 后续处理与 ask() 类似
```

## 4.8 支持的模型类型

```python
# 推理模型（特殊参数处理）
REASONING_MODELS = ["o1", "o3-mini"]

# 多模态模型（支持图像输入）
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]
```

## 4.9 Bedrock 客户端

**文件位置**：`app/bedrock.py`

```python
class BedrockClient:
    """AWS Bedrock 客户端封装"""

    def __init__(self):
        self.boto_client = boto3.client("bedrock-runtime")

    async def chat_completions_create(self, **params):
        """Bedrock API 调用"""
        model = params.pop("model")
        messages = params.pop("messages")

        # 转换为 Bedrock 格式
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": params.get("max_tokens", 4096),
            "messages": messages,
        }

        response = self.boto_client.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        return json.loads(response["body"].read())
```

## 4.10 重试机制

```python
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(
    wait=wait_random_exponential(min=1, max=60),  # 指数退避等待
    stop=stop_after_attempt(6),                   # 最多重试6次
    retry=retry_if_exception_type(
        (OpenAIError, Exception, ValueError)
    ),
)
async def ask(...):
    """带重试的请求方法"""
    pass
```

## 4.11 错误处理

```python
# Token 限制异常
class TokenLimitExceeded(Exception):
    """Token 限制超出"""
    pass

# 处理不同类型的 API 错误
try:
    response = await self.client.chat.completions.create(...)
except AuthenticationError:
    logger.error("认证失败，请检查 API Key")
except RateLimitError:
    logger.error("速率限制超出，考虑增加重试次数")
except APIError as e:
    logger.error(f"API 错误: {e}")
except Exception:
    logger.exception("未知错误")
```
