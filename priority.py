import os
import asyncio
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from config import Config
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

llm = ChatOpenAI(
    temperature=0.5,
    model=Config.LLM_MODEL,
    api_key=Config.API_KEY,
    base_url=Config.BASE_URL,
)
# 1. 任务管理系统 —— Pydantic 模型 + 内存管理器
class Task(BaseModel):
    """表示系统中的单个任务（Pydantic 保证字段类型安全）。"""
    id: str
    description: str
    priority: Optional[str] = None
    assigned_to: Optional[str] = None

class SuperSimpleTaskManager:
    """高效且稳健的内存任务管理器。"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.next_task_id = 1

    def create_task(self, description: str) -> Task:
        """创建并存储一个新任务。"""
        task_id = f"Task_{self.next_task_id}"
        new_task = Task(id=task_id, description=description)
        self.tasks[task_id] = new_task
        self.next_task_id += 1
        print(f"DEBUG: 任务已创建 - {task_id}: {description}")
        return new_task

    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        #新建：它会复制原对象的所有字段，然后用 update_data 中的值覆盖对应的字段，生成一个新对象。
        """使用 Pydantic 的 model_copy 安全地更新任务。"""
        task = self.tasks.get(task_id)
        if task:
            update_data = {k: v for k, v in kwargs.items() if v is not None}
            updated_task = task.model_copy(update=update_data)
            self.tasks[task_id] = updated_task
            print(f"DEBUG: 任务 {task_id} 已更新为 {update_data}")
            return updated_task
        print(f"DEBUG: 未找到任务 {task_id} 进行更新。")
        return None

    def list_all_tasks(self) -> str:
        """列出系统中当前的所有任务。"""
        if not self.tasks:
            return "当前没有任务。"
        task_strings = []
        for task in self.tasks.values():
            task_strings.append(
                f"ID:{task.id}, 描述:{task.description}, "
                f"优先级: {task.priority or 'N/A'}, "
                f"分配给: {task.assigned_to or 'N/A'}"
            )
        return "当前任务:\n" + "\n".join(task_strings)
task_manager = SuperSimpleTaskManager()

# 2. 工具封装 —— Pydantic args_schema 做参数校验
class CreateTaskArgs(BaseModel):
    description: str = Field(description="任务的详细描述。")

class PriorityArgs(BaseModel):
    task_id: str = Field(description="要更新的任务 ID，例如 'Task_1'。")
    priority: str = Field(description="要分配任务的优先级（P0、P1、P2）。")

class AssignWorkerArgs(BaseModel):
    task_id: str = Field(description="要更新的任务 ID，例如 'Task_1'。")
    worker_name: str = Field(description="要分配任务的工作人员的名字。")

def create_new_task_tool(description: str) -> str:
    """使用给定的描述创建一个新的项目任务。"""
    task = task_manager.create_task(description)
    return f"任务已创建 {task.id}: {task.description}"

def assign_priority_to_task_tool(task_id: str, priority: str) -> str:
    """为给定的任务 ID 分配优先级（P0、P1、P2）。"""
    if priority not in ["P0", "P1", "P2"]:
        return "优先级无效。必须是 P0、P1 或 P2。"
    task = task_manager.update_task(task_id, priority=priority)
    return (
        f"已为任务 {task.id} 分配优先级 {priority}。" if task
        else f"未找到任务 {task_id}。"
    )

def assign_task_to_worker_tool(task_id: str, worker_name: str) -> str:
    """将任务分配给特定的工作人员。"""
    task = task_manager.update_task(task_id, assigned_to=worker_name)
    return (
        f"已将任务 {task.id} 分配给 {worker_name}。" if task
        else f"未找到任务 {task_id}。"
    )

# PM 智能体使用的所有工具
pm_tools = [
    create_new_task_tool,
    assign_priority_to_task_tool,
    assign_task_to_worker_tool,
    task_manager.list_all_tasks,
]


# 3. 系统提示词
pm_system_prompt = (
    "你是一个专注的项目管理 LLM 智能体。你的目标是高效地管理项目任务。\n"
    "当你收到新的任务请求时，遵循以下步骤：\n"
    "1. 首先，使用 `create_new_task_tool` 工具创建具有给定描述的任务。"
    "你必须首先执行此操作以获取 `task_id`。\n"
    "2. 接下来，分析用户的请求以查看是否提到了优先级或受让人。\n"
    "   - 如果提到优先级（例如，'紧急'、'ASAP'、'关键'），将其映射到 P0。"
    "使用 `assign_priority_to_task_tool`。\n"
    "   - 如果提到工作人员，使用 `assign_task_to_worker_tool`。\n"
    "3. 如果缺少任何信息（优先级、受让人），"
    "你必须做出合理的默认分配（例如，分配 P1 优先级并分配给 'Worker A'）。\n"
    "4. 一旦任务完全处理完毕，使用 `list_all_tasks` 显示最终状态。\n\n"
    "可用的工作人员：'Worker A'、'Worker B'、'Review Team'\n"
    "优先级级别：P0（最高）、P1（中等）、P2（最低）"
)

# 4. 创建 Agent
pm_agent = create_agent(
    model=llm,
    tools=pm_tools,
    system_prompt=pm_system_prompt,
)

# 5. 模拟交互
async def run_simulation():
    print("--- 项目管理模拟 ---\n")

    # 场景 1：处理新的紧急功能请求
    print("[用户请求] 我需要尽快实现一个新的登录系统。它应该分配给 Worker B。\n")
    result = await pm_agent.ainvoke({
        "messages": [
            {"role": "user", "content": "创建一个实现新登录系统的任务。这很紧急，应该分配给 Worker B。"}
        ]
    })
    # 打印 LLM 最终回复
    final_msg = result["messages"][-1]
    print(f"\n[Agent 回复] {final_msg.content}")
    print("\n" + "-" * 60 + "\n")

    # 场景 2：处理细节较少的不太紧急的内容更新
    print("[用户请求] 我们需要审查营销网站内容。\n")
    result = await pm_agent.ainvoke({
        "messages": [
            {"role": "user", "content": "管理一个新任务：审查营销网站内容。"}
        ]
    })
    final_msg = result["messages"][-1]
    print(f"\n[Agent 回复] {final_msg.content}")
    print("\n--- 模拟完成 ---")


if __name__ == "__main__":
    asyncio.run(run_simulation())
