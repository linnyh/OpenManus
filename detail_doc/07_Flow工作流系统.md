# 第七章：Flow工作流系统

## 7.1 Flow 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      BaseFlow                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  agents: Dict[str, BaseAgent]                        │  │
│  │  primary_agent: BaseAgent                            │  │
│  │  executor_keys: List[str]                           │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │PlanningFlow │ │  ParallelFlow │ │  Sequential  │
    │  (规划流)   │ │  (并行流)    │ │    Flow     │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## 7.2 BaseFlow 基类

**文件位置**：`app/flow/base.py`

```python
class BaseFlow(BaseModel, ABC):
    """工作流基类"""

    agents: Dict[str, BaseAgent] = Field(default_factory=dict)
    primary_agent: Optional[BaseAgent] = None
    executor_keys: List[str] = Field(default_factory=list)

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """执行工作流"""
        pass

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """获取执行器 Agent"""
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        return self.primary_agent
```

## 7.3 PlanningFlow 实现

**文件位置**：`app/flow/planning.py`

### 7.3.1 核心属性

```python
class PlanningFlow(BaseFlow):
    """基于 LLM 规划的多步骤工作流"""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
```

### 7.3.2 步骤状态枚举

```python
class PlanStepStatus(str, Enum):
    """计划步骤状态"""

    NOT_STARTED = "not_started"     # 未开始
    IN_PROGRESS = "in_progress"    # 执行中
    COMPLETED = "completed"        # 已完成
    BLOCKED = "blocked"            # 被阻塞

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """获取活跃状态列表"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """获取状态标记"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }
```

### 7.3.3 执行主流程

```python
async def execute(self, input_text: str) -> str:
    """执行规划工作流"""
    try:
        if not self.primary_agent:
            raise ValueError("No primary agent available")

        # 1. 创建初始计划
        if input_text:
            await self._create_initial_plan(input_text)

            if self.active_plan_id not in self.planning_tool.plans:
                return f"Failed to create plan for: {input_text}"

        result = ""

        # 2. 执行循环
        while True:
            # 获取当前步骤
            self.current_step_index, step_info = await self._get_current_step_info()

            # 无更多步骤，退出循环
            if self.current_step_index is None:
                result += await self._finalize_plan()
                break

            # 获取执行器
            step_type = step_info.get("type") if step_info else None
            executor = self.get_executor(step_type)

            # 执行步骤
            step_result = await self._execute_step(executor, step_info)
            result += step_result + "\n"

            # 检查是否终止
            if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                break

        return result

    except Exception as e:
        logger.error(f"Error in PlanningFlow: {str(e)}")
        return f"Execution failed: {str(e)}"
```

### 7.3.4 创建初始计划

```python
async def _create_initial_plan(self, request: str) -> None:
    """基于请求创建初始计划"""
    logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

    # 构建系统消息
    system_message_content = (
        "You are a planning assistant. Create a concise, actionable plan with clear steps. "
        "Focus on key milestones rather than detailed sub-steps. "
        "Optimize for clarity and efficiency."
    )

    # 添加 Agent 描述
    agents_description = []
    for key in self.executor_keys:
        if key in self.agents:
            agents_description.append({
                "name": key.upper(),
                "description": self.agents[key].description,
            })

    if len(agents_description) > 1:
        system_message_content += (
            f"\nAvailable agents: {json.dumps(agents_description)}\n"
            "When creating steps, specify agent names using format '[agent_name]'."
        )

    # 调用 LLM 创建计划
    system_message = Message.system_message(system_message_content)
    user_message = Message.user_message(
        f"Create a reasonable plan with clear steps to accomplish: {request}"
    )

    response = await self.llm.ask_tool(
        messages=[user_message],
        system_msgs=[system_message],
        tools=[self.planning_tool.to_param()],
        tool_choice=ToolChoice.AUTO,
    )

    # 处理工具调用
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.function.name == "planning":
                args = json.loads(tool_call.function.arguments)
                args["plan_id"] = self.active_plan_id
                result = await self.planning_tool.execute(**args)
                logger.info(f"Plan creation result: {str(result)}")
                return

    # 默认计划
    logger.warning("Creating default plan")
    await self.planning_tool.execute(**{
        "command": "create",
        "plan_id": self.active_plan_id,
        "title": f"Plan for: {request[:50]}...",
        "steps": ["Analyze request", "Execute task", "Verify results"],
    })
```

### 7.3.5 获取当前步骤

```python
async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
    """获取当前待执行的步骤"""
    if not self.active_plan_id or self.active_plan_id not in self.planning_tool.plans:
        return None, None

    plan_data = self.planning_tool.plans[self.active_plan_id]
    steps = plan_data.get("steps", [])
    step_statuses = plan_data.get("step_statuses", [])

    # 查找第一个未完成的步骤
    for i, step in enumerate(steps):
        if i >= len(step_statuses):
            status = PlanStepStatus.NOT_STARTED.value
        else:
            status = step_statuses[i]

        if status in PlanStepStatus.get_active_statuses():
            step_info = {"text": step}

            # 提取步骤类型
            import re
            type_match = re.search(r"\[([A-Z_]+)\]", step)
            if type_match:
                step_info["type"] = type_match.group(1).lower()

            # 标记为进行中
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=i,
                step_status=PlanStepStatus.IN_PROGRESS.value,
            )

            return i, step_info

    return None, None
```

### 7.3.6 执行单个步骤

```python
async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
    """执行单个步骤"""
    plan_status = await self._get_plan_text()
    step_text = step_info.get("text", f"Step {self.current_step_index}")

    # 构建执行提示
    step_prompt = f"""
    CURRENT PLAN STATUS:
    {plan_status}

    YOUR CURRENT TASK:
    You are now working on step {self.current_step_index}: "{step_text}"

    Please only execute this current step using the appropriate tools.
    When you're done, provide a summary of what you accomplished.
    """

    try:
        # 使用 Agent 执行
        step_result = await executor.run(step_prompt)

        # 标记为完成
        await self._mark_step_completed()

        return step_result
    except Exception as e:
        logger.error(f"Error executing step {self.current_step_index}: {e}")
        return f"Error executing step {self.current_step_index}: {str(e)}"
```

### 7.3.7 标记步骤完成

```python
async def _mark_step_completed(self) -> None:
    """标记当前步骤为已完成"""
    if self.current_step_index is None:
        return

    try:
        await self.planning_tool.execute(
            command="mark_step",
            plan_id=self.active_plan_id,
            step_index=self.current_step_index,
            step_status=PlanStepStatus.COMPLETED.value,
        )
    except Exception as e:
        logger.warning(f"Failed to update plan status: {e}")
        # 直接更新存储
        if self.active_plan_id in self.planning_tool.plans:
            plan_data = self.planning_tool.plans[self.active_plan_id]
            step_statuses = plan_data.get("step_statuses", [])

            while len(step_statuses) <= self.current_step_index:
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)

            step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
            plan_data["step_statuses"] = step_statuses
```

### 7.3.8 计划完成总结

```python
async def _finalize_plan(self) -> str:
    """完成计划并提供总结"""
    plan_text = await self._get_plan_text()

    try:
        system_message = Message.system_message(
            "You are a planning assistant. Summarize the completed plan."
        )
        user_message = Message.user_message(
            f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\n"
            "Please provide a summary of what was accomplished."
        )

        response = await self.llm.ask(messages=[user_message], system_msgs=[system_message])
        return f"Plan completed:\n\n{response}"

    except Exception as e:
        logger.error(f"Error finalizing plan: {e}")
        return "Plan completed. Error generating summary."
```

## 7.4 FlowFactory 工厂

**文件位置**：`app/flow/flow_factory.py`

```python
class FlowType(str, Enum):
    """Flow 类型枚举"""
    PLANNING = "planning"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

class FlowFactory:
    """Flow 创建工厂"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs
    ) -> BaseFlow:
        """创建指定类型的 Flow"""

        # 标准化 agents 参数
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        # 根据类型创建 Flow
        if flow_type == FlowType.PLANNING:
            return PlanningFlow(agents=agents_dict, **kwargs)
        elif flow_type == FlowType.PARALLEL:
            return ParallelFlow(agents=agents_dict, **kwargs)
        elif flow_type == FlowType.SEQUENTIAL:
            return SequentialFlow(agents=agents_dict, **kwargs)
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
```

## 7.5 PlanningTool 计划工具

**文件位置**：`app/tool/planning.py`

```python
class PlanningTool(BaseTool):
    """计划管理工具"""

    name: str = "planning"
    description: str = "Create and manage execution plans"

    plans: Dict[str, dict] = {}  # plan_id -> plan_data

    async def execute(
        self,
        command: str,
        plan_id: str,
        **kwargs
    ) -> ToolResult:
        """执行计划命令"""
        if command == "create":
            return await self._create_plan(plan_id, **kwargs)
        elif command == "get":
            return await self._get_plan(plan_id)
        elif command == "mark_step":
            return await self._mark_step(plan_id, **kwargs)
        elif command == "update":
            return await self._update_plan(plan_id, **kwargs)
```

## 7.6 Flow 执行流程图

```
用户输入
    │
    ▼
┌────────────────────────────────────────────┐
│  PlanningFlow.execute()                    │
└─────────────────┬────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  _create_initial_plan()                    │
│  - LLM 生成计划                             │
│  - planning_tool.execute()                │
│  - 存储到 planning_tool.plans              │
└─────────────────┬────────────────────────┘
                   │
                   ▼
            ┌────────────┐
            │  循环开始   │
            └─────┬──────┘
                  │
                  ▼
┌────────────────────────────────────────────┐
│  _get_current_step_info()                  │
│  - 获取第一个 NOT_STARTED/IN_PROGRESS 步骤  │
│  - 提取步骤类型                             │
│  - 标记为 IN_PROGRESS                       │
└─────────────────┬────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  步骤 == None? │
         └───────┬────────┘
                 │
    ┌────────────┴────────────┐
    │ 是                     │ 否
    ▼                        ▼
┌─────────┐           ┌──────────────────┐
│_finalize│           │ _execute_step()   │
│ _plan() │           │ - 构建执行提示     │
└────┬────┘           │ - executor.run()  │
     │                │ - 标记为 COMPLETED │
     │                └────────┬───────────┘
     │                         │
     │                         ▼
     │              ┌─────────────────────┐
     │              │ Agent 执行中...    │
     │              └──────────┬─────────┘
     │                         │
     │                         ▼
     │              ┌─────────────────────┐
     │              │ 状态 == FINISHED?   │
     │              └──────────┬─────────┘
     │                         │
     │           ┌─────────────┴─────────────┐
     │           │ 是                       │ 否
     │           ▼                          ▼
     │    ┌──────────┐              ┌──────────────┐
     │    │  退出循环 │              │  继续循环    │
     │    └──────────┘              └──────┬───────┘
     │                                     │
     └─────────────────────────────────────┘
                  │
                  ▼
          ┌─────────────────┐
          │  返回执行结果   │
          └─────────────────┘
```

## 7.7 使用示例

```python
from app.agent.manus import Manus
from app.agent.data_analysis import DataAnalysis
from app.flow.flow_factory import FlowFactory, FlowType

async def run_flow():
    # 定义多个 Agent
    agents = {
        "manus": Manus(),
        "data_analysis": DataAnalysis(),
    }

    # 创建 PlanningFlow
    flow = FlowFactory.create_flow(
        flow_type=FlowType.PLANNING,
        agents=agents,
    )

    # 执行工作流
    prompt = input("Enter your request: ")
    result = await flow.execute(prompt)

    print(result)

if __name__ == "__main__":
    asyncio.run(run_flow())
```
