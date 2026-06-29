# 🤖 AI Agent 学习项目 — 从零到生产级

> 一份精心编排的 AI Agent 学习路线，涵盖 LangGraph、CrewAI 两大框架，从最基础的"图"概念一路进阶到 Plan-and-Execute、Human-in-the-Loop、Agent-as-a-Service 等生产级模式。

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-green)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.15-orange)](https://docs.langchain.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-latest-purple)](https://docs.crewai.com/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen--Max-red)](https://dashscope.aliyun.com/)

---

## 📖 目录

- [项目概述](#-项目概述)
- [学习路线图](#-学习路线图)
- [快速开始](#-快速开始)
- [第零层：环境与工具基础](#第零层环境与工具基础)
- [第一层：LangGraph 核心概念](#第一层langgraph-核心概念)
- [第二层：状态持久化与记忆](#第二层状态持久化与记忆)
- [第三层：流程控制模式](#第三层流程控制模式)
- [第四层：Human-in-the-Loop 人机协作](#第四层human-in-the-loop-人机协作)
- [第五层：多 Agent 协作与路由](#第五层多-agent-协作与路由)
- [第六层：复杂工作流编排](#第六层复杂工作流编排)
- [第七层：CrewAI 框架](#第七层crewai-框架)
- [第八层：生产级模式](#第八层生产级模式)
- [核心概念速查](#-核心概念速查)
- [技术栈](#-技术栈)

---

## 📋 项目概述

这个仓库记录了我从零学习 AI Agent 开发的完整过程。目标是**理解原理而非死记 API**——每个文件都聚焦于一个独立的设计模式或架构思想。

### 什么是 AI Agent？

AI Agent 不是普通的"一问一答"聊天机器人。它是一个**能自主决策、调用工具、多步推理**的智能体：

```
普通 LLM 调用：  用户问 → LLM 答 → 结束
AI Agent：       用户问 → LLM 分析 → 调用工具 → 分析结果 → 再思考 → ... → 给出最终答案
```

### 为什么选 LangGraph？

LangGraph 用一个统一的隐喻来建模所有 Agent 行为：**有向图（Graph）**。

```python
# 整个 Agent 就是一个图：节点做事，边决定方向
StateGraph(State)
  .add_node("think", call_llm)      # 节点 = 做事的函数
  .add_node("act", execute_tools)   # 节点 = 做事的函数
  .add_edge("think", "act")         # 边 = 顺序流转
  .add_conditional_edges("think", decide_next)  # 条件边 = 决策分支
```

这个简单模型能表达几乎所有 Agent 架构——从简单的 ReAct 循环到复杂的多 Agent 协作。

---

## 🗺️ 学习路线图

```
                        ┌─────────────────────────────────┐
                        │  🤖 langgraph_base.py           │
                        │  第一个 Agent：图 = 节点+边+状态  │
                        │  "Agent 的 Hello World"          │
                        └─────────────┬───────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼              ▼
   状态持久化      流程控制        Human-in-Loop   多Agent协作    子图/计划      生产模式
   checkpointer   fallback        Interrupt_test  multi-router   subgraph      research_choose
   memory1        send_case       react_case      reflect        agent.workflow a2a
   (记忆)         (降级+并行)     human-in-loop   CrewAI系列      (Plan-Execute) guardrails
                                                                                llm_as_judge
                                                                                priority
```

**推荐按以下顺序阅读代码：**

| 层级 | 文件 | 学什么 | 难度 |
|------|------|--------|------|
| 0️⃣ | `config.py`, `tool.py` | 环境搭建、工具定义 | ⭐ |
| 1️⃣ | `langgraph_base.py` | 图、节点、边、条件路由、checkpointer | ⭐⭐ |
| 2️⃣ | `checkpointer_case.py`, `memory1.py` | 持久化存储、对话记忆 | ⭐⭐ |
| 3️⃣ | `fallback.py`, `send_case.py` | 降级策略、并行 fan-out | ⭐⭐⭐ |
| 4️⃣ | `Interrupt _test.py`, `react_case.py`, `human-in-the-loop.py` | 人机协作、中断恢复 | ⭐⭐⭐ |
| 5️⃣ | `multi-router1.py`, `multi-router2.py`, `reflect.py` | 路由、多 Agent、自我反思 | ⭐⭐⭐⭐ |
| 6️⃣ | `subgraph.py`, `subgraph_stream.py`, `agent.workflow.py` | 子图嵌套、Plan-and-Execute | ⭐⭐⭐⭐ |
| 7️⃣ | `planning.py`, `mutil.py`, `tool_crewai.py` | CrewAI 角色建模 | ⭐⭐ |
| 8️⃣ | `research_choose.py`, `a2a.py`, `guardrails.py`, `priority.py`, `llm_as_judge.py` | 模型路由、A2A 协议、安全护栏、LLM 评审 | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速开始

### 环境要求

```bash
# Python 3.10+
# 创建虚拟环境
python -m venv lg_env
source lg_env/bin/activate   # Linux/Mac
lg_env\Scripts\activate      # Windows

# 安装依赖
pip install langgraph langchain langchain-openai langchain-tavily crewai
```

### 配置

创建 `.env` 文件（已加入 `.gitignore`）：

```env
API_KEY=your_api_key_here
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
TAVILY_API_KEY=your_tavily_key
```

> 本项目默认使用阿里云百炼 Qwen-Max 模型（通过 OpenAI 兼容接口）。你也可以换成任何 OpenAI 兼容的模型（Groq、DeepSeek 等）。

### 运行第一个 Agent

```bash
python langgraph_base.py
# 预期输出: "现在30度，有雾"（模拟搜索上海天气）
```

---

## 第零层：环境与工具基础

### 📄 [config.py](config.py) — 统一配置

```python
class Config:
    API_KEY   = os.getenv("API_KEY")
    BASE_URL  = os.getenv("BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
```

**设计思路：** 所有文件共用一个配置类，避免 API Key 硬编码和散落各处。切换模型只需改 `.env`，无需改代码。

### 📄 [tool.py](tool.py) — 最简单的 LangChain Agent

这是你能写出的**最短的 Agent**。演示了 LangChain 经典 API 的三个核心概念：

```python
# 1. 定义工具
@tool
def search_information(query: str) -> str: ...

# 2. 创建 Agent（LLM + 工具 + prompt）
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 3. 并发执行多个任务
await asyncio.gather(*tasks)
```

**关键洞察：** `AgentExecutor` 是 LangChain 经典版的"Agent 运行时"——它负责调用 LLM → 解析工具调用 → 执行工具 → 把结果传回 LLM 的循环。LangGraph 后来用"图"替代了这个运行时，但思想完全一致。

---

## 第一层：LangGraph 核心概念

### 📄 [langgraph_base.py](langgraph_base.py) — ⭐ 最重要的文件

这是整个学习之旅的基石。如果你只能彻底理解一个文件，就应该是这个。

#### 它实现了什么？

一个**能调用工具的 Agent 循环**。当用户问"上海天气怎么样"，Agent 会自动调用搜索工具获取数据，然后基于工具返回的结果生成回答。

#### 架构图

```
         ┌──────────┐
         │   START   │
         └────┬─────┘
              ▼
         ┌─────────┐
    ┌───▶│  agent  │ (LLM 分析问题)
    │    └────┬────┘
    │         │ should_continue()
    │    ┌────┴──────┐
    │    ▼            ▼
    │ ┌──────┐    ┌──────┐
    │ │ tools│    │ END  │
    │ └──┬───┘    └──────┘
    │    │
    └────┘
```

#### 五个核心概念

**1. State（状态）** — 流转的数据

```python
workflow = StateGraph(MessagesState)
# MessagesState 内置一个 'messages' 列表，新消息自动追加到旧消息后面
```

**2. Node（节点）** — 做事的函数

```python
def call_model(state):
    response = model.invoke(state['messages'])
    return {"messages": [response]}  # 返回值追加到 state.messages 末尾
```

**3. Edge（边）** — 流程方向

```python
workflow.add_edge("tools", "agent")  # tools 执行完 → 回到 agent
```

**4. Conditional Edge（条件边）** — 决策分支

```python
def should_continue(state):
    if state['messages'][-1].tool_calls:
        return "tools"   # LLM 要调工具 → 去 tools 节点
    return END           # 否则结束

workflow.add_conditional_edges("agent", should_continue)
```

**5. Checkpointer（检查点）** — 记忆

```python
app = workflow.compile(checkpointer=MemorySaver())
# thread_id 区分不同对话
app.invoke(..., config={"configurable": {"thread_id": 42}})
```

#### 💡 关键洞察

- `bind_tools(tools)` 是告诉 LLM"你可以用这些工具"——本质是在 API 调用中传递 `tools` 参数
- `should_continue` 是整张图的**交通警察**——它不看消息内容，只看 LLM 返回的消息里有没有 `tool_calls`
- `thread_id` 是你对话记忆的"钥匙"——不同的 thread_id 完全隔离，就像一个服务台同时服务多个互不干扰的客户

---

## 第二层：状态持久化与记忆

### 📄 [checkpointer_case.py](checkpointer_case.py) — SQLite 持久化

```python
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
    graph = builder.compile(checkpointer=memory)
    result = await graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
```

**对比：**

| 类型 | 存储位置 | 生命周期 | 适用场景 |
|------|---------|---------|---------|
| `MemorySaver` | 内存 | 进程结束就没了 | 开发调试 |
| `AsyncSqliteSaver` | SQLite 文件 | 永久 | 生产环境 |

### 📄 [memory1.py](memory1.py) — LangChain 经典版记忆

```python
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

**设计思路：** 展示 LangChain 经典版的记忆方案——`ConversationBufferMemory` 自动管理对话历史。与 LangGraph 的 checkpointer 是两种思路：前者是"对话记忆"，后者是更通用的"状态快照"。

---

## 第三层：流程控制模式

### 📄 [fallback.py](fallback.py) — 降级策略

**场景：** 精确查询失败 → 自动切换到模糊查询，而不是直接报错。

```
用户查询 → primary_handler(精确位置) ─成功→ respond(返回结果)
                  │
                  └失败→ fallback_handler(粗略位置) → respond
```

```python
builder.add_conditional_edges(
    "primary",
    lambda s: "fallback" if s.get("primary_location_failed") else "respond"
)
```

**生产价值：** 这就是微服务里的 Circuit Breaker 和 Fallback 模式。Agent 不应该把原始错误抛给用户。

### 📄 [send_case.py](send_case.py) — 并行 Fan-out

**场景：** 一个任务需要对多个对象同时做同样的事。

```python
# 输入: ["cats", "dogs"] → 同时生成两个笑话 → 合并结果
def continue_to_jokes(state):
    return [Send("generate_joke", {"subject": s}) for s in state["subject"]]
```

**核心原理：** `Send` 为每个元素创建一个独立的任务实例，并行执行。`operator.add` 让所有结果自动合并到同一个列表里。

---

## 第四层：Human-in-the-Loop 人机协作

这是 LangGraph 最强大的特性之一：**让人类在 Agent 决策的关键节点介入审批**。

### 📄 [Interrupt _test.py](Interrupt_test.py) — 基础中断机制

```python
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])
# 图跑到 step_3 之前自动暂停
```

```
step_1 → step_2 → ⏸️ [等待人类确认] → step_3 或取消
```

恢复运行：
```python
graph.stream(None, thread, stream_mode="values")  # None = 从断点继续，不传新数据
```

### 📄 [react_case.py](react_case.py) — ReAct Agent + 工具调用拦截

```python
graph = create_agent(model, tools=tools, interrupt_before=["tools"], checkpointer=memory)
# Agent 每次想调用工具前，必须先经过人类批准
```

**安全价值：** Agent 想删除数据库？先弹出确认框。想发送邮件？先让你预览内容。这是 AI 安全的最后一道防线。

### 📄 [human-in-the-loop.py](human-in-the-loop.py) — 完整案例

一个**技术支持客服 Agent**，集成了：
- 客户个性化（根据等级/购买历史定制回复）
- 工具调用拦截（需人工确认才执行）
- 多种工具选择（故障排查、创建工单、转接人工）

```python
user_input = input("\n是否执行此工具？(y/n): ").strip().lower()
if user_input == 'y':
    result = troubleshoot_issue.invoke(tool_args)
else:
    final = "您已取消工具调用..."
```

---

## 第五层：多 Agent 协作与路由

当一个 Agent 不够时，你让多个 Agent 各司其职。

### 📄 [multi-router1.py](multi-router1.py) — RunnableBranch 路由

使用 LangChain 的 `RunnableBranch`，根据意图分类结果自动路由到不同的处理函数：

```
用户请求 → 协调器 LLM → booker? → booking_handler
                       → info?   → info_handler
                       → unclear → unclear_handler
```

### 📄 [multi-router2.py](multi-router2.py) — Function Calling 路由

同一功能但用**原生 OpenAI API** 实现，展示在 LangChain 封装之下到底发生了什么：

```python
class Coordinator:
    def process_request(self, user_request):
        intention = self._classify_intent(user_request)  # LLM 判断意图
        if intention == "booking":
            return self.booking_agent.process(user_request)
        elif intention == "info":
            return self.info_agent.process(user_request)
```

**学习建议：** 两个文件对比着看。`multi-router1.py` 告诉你"用 LangChain 怎么写"，`multi-router2.py` 告诉你"LangChain 在背后做了什么"。

### 📄 [reflect.py](reflect.py) — 自我反思循环

让 LLM 切换角色审查自己生成的内容：

```
生成初始代码 → 反思（扮演高级代码审查员）
   ├─ CODE_IS_PERFECT → 结束
   └─ 有毛病 → 带着批评重新生成 → 再反思 → ...
```

**核心技巧：** 反思阶段用**完全不同的 System Prompt**（"你是一名高级软件工程师和 Python 专家，执行细致代码审查"），让同一个 LLM 扮演不同的角色。

---

## 第六层：复杂工作流编排

### 📄 [subgraph.py](subgraph.py) — 子图嵌套

当单个图过于复杂时，把子流程封装成独立图，在父图中像普通节点一样使用：

```
父图: convert_logs → ┬→ failure_analysis（子图1）→ ┐
                      └→ question_summarization（子图2）→ ┴→ 合并结果（operator.add）
```

**关键设计：**
- 子图有自己的**独立状态类型**（`FailureAnalysisState`, `QuestionSummaryState`）
- 父图通过适配器函数将子图结果**映射**回父图状态
- 两条并行支路的结果通过 `operator.add` **自动合并**

### 📄 [subgraph_stream.py](subgraph_stream.py) — 子图 + 流式输出

在 `subgraph.py` 基础上加上：
- **异步执行**（`async/await` 模拟真实 IO）
- **流式输出**（`stream_mode="values"` 实时展示状态变化）
- **状态持久化**（`MemorySaver`）

### 📄 [agent.workflow.py](agent.workflow.py) — ⭐ Plan-and-Execute 模式

这是整个项目里最复杂的 Agent 架构之一，灵感来自 [Plan-and-Solve](https://arxiv.org/abs/2305.04091) 论文。

#### 核心思想

```
START → plan(制定计划) → agent(执行第一步) → replan(重新评估)
                                  ↑               │
                                  └─── 还有步骤 ───┘
                                                    │
                                              可以结束了?
                                                    ▼
                                                  END
```

**为什么需要 replan？** 执行第一步后可能发现新信息，使原计划需要调整。这不是死板地按计划走，而是**动态调整**。

#### 状态设计

```python
class PlanExecute(TypedDict):
    input: str                               # 用户原始问题
    plan: List[str]                          # 当前计划（逐步被消耗）
    past_steps: Annotated[List[Tuple], operator.add]  # 已完成步骤（追加）
    response: str                            # 最终回答
```

#### 💡 三个关键修复（踩坑记录）

1. **Qwen 模型在使用结构化输出时，prompt 中必须包含 "JSON" 关键词**，否则 API 返回 400。
2. **`max_results` 不能设太小**——Tavily 免费版返回质量不稳定，`max_results=3` 更可靠。
3. **System Prompt 必须明确要求"先搜索再回答"**，否则模型可能凭过时的训练数据猜测。

---

## 第七层：CrewAI 框架

CrewAI 是 LangGraph 的补充——用**角色建模**（`role`/`goal`/`backstory`）来定义 Agent 行为，更接近"人"的隐喻。

| 文件 | 模式 | 说明 |
|------|------|------|
| [planning.py](planning.py) | 单 Agent 规划 | 一个 Agent 先规划再写作 |
| [mutil.py](mutil.py) | 多 Agent 顺序协作 | 研究员→作家流水线 |
| [tool_crewai.py](tool_crewai.py) | 自定义工具 | 股票查询工具 + 财务分析师 Agent |

```python
researcher = Agent(
    role='高级研究分析师',
    goal='查找并总结 AI 的最新趋势',
    backstory='你是一位经验丰富的研究分析师...'
)
writer = Agent(
    role='技术内容作家',
    goal='撰写引人入胜的博客文章',
    backstory='你是一位熟练的作家...'
)
```

**LangGraph vs CrewAI：** LangGraph 提供精细化流程控制（图、边、条件），CrewAI 提供高层角色抽象。实际项目经常混用——用 CrewAI 定义 Agent 角色，用 LangGraph 编排它们之间的协作流程。

---

## 第八层：生产级模式

### 📄 [research_choose.py](research_choose.py) — 智能模型路由

**场景：** 不同复杂度的问题用不同能力的模型，优化成本和速度。

```
用户提问 → 分类器 → simple?         → Llama 4 Scout (17B · 快 · 便宜)
                   → reasoning?      → Llama 3.3 (70B · 推理强)
                   → internet_search → GPT-OSS (120B) + Tavily 联网搜索
```

**核心价值：** 简单事实查询（"法国首都是什么"）不浪费大模型资源，复杂推理才上大模型。这就是**模型调度优化**。

### 📄 [a2a.py](a2a.py) — Agent-as-a-Service (A2A 协议)

把 Agent 封装成 FastAPI Web 服务，对外暴露标准 API：

| 端点 | 功能 |
|------|------|
| `GET /.well-known/agent-card` | Agent 名片（告诉世界"我能做什么"） |
| `POST /tasks` | 创建异步任务 |
| `GET /tasks/{task_id}` | 查询任务状态 |
| `POST /sendTask` | 同步请求 |
| `POST /sendTaskSubscribe` | **SSE 流式响应** |

**这是把 Agent 部署到生产环境的完整方案**——你的 Agent 不再是一个本地脚本，而是可以被其他团队通过 HTTP 调用的微服务。

### 📄 [guardrails.py](guardrails.py) — 安全护栏

在 Agent 处理用户输入**之前**，用另一个 LLM 进行安全检查：

```python
# 输入 → 安全策略评估 → 合规 ✅ → 主 Agent 处理
#                     → 违规 ❌ → 拦截！
```

检测：越狱尝试、仇恨言论、危险活动、露骨材料、学术不诚实。**这就是 OpenAI Moderation API 的自建版本。**

### 📄 [priority.py](priority.py) — 项目管理 Agent

演示如何用 AI Agent 管理实际工作：自动创建任务、分配优先级、指派人员。

```python
@tool
def assign_priority_to_task_tool(task_id: str, priority: str) -> str:
    """为给定的任务 ID 分配优先级（P0、P1、P2）"""
```

**亮点：** 用 **Pydantic BaseModel** 定义工具参数 schema，获得自动类型校验。

### 📄 [llm_as_judge.py](llm_as_judge.py) — LLM-as-Judge

让 LLM 扮演评审角色，对内容质量打分：

```python
# 评分维度：清晰性、中立性、相关性、完整性、受众适当性（各1-5分）
{
  "overall_score": 4,
  "rationale": "...",
  "detailed_feedback": ["...", "..."]
}
```

**应用场景：** 内容审核、RAG 检索质量评估、模型输出打分——所有需要"让 AI 评价 AI"的场景。

---

## 🧠 核心概念速查

| 概念 | LangGraph 实现 | 一句话解释 |
|------|---------------|-----------|
| **State** | `StateGraph(MyState)` | 贯穿全图的数据总线，所有节点读/写它 |
| **Node** | `add_node("name", fn)` | 做事情的函数，读 State、返回新 State |
| **Edge** | `add_edge("A", "B")` | A 完成后必定去 B |
| **Conditional Edge** | `add_conditional_edges("A", fn, mapping)` | A 完成后根据 fn 的返回值选择去向 |
| **Checkpointer** | `compile(checkpointer=...)` | 状态快照存储器 = Agent 的"记忆" |
| **Tool** | `@tool` + `bind_tools()` | 给 LLM 提供"手"，让它能执行实际操作 |
| **Interrupt** | `interrupt_before=["tools"]` | 在指定节点前暂停，等人类审批 |
| **Send** | `Send("node", data)` | 并行 fan-out，一个任务分裂成 N 个 |
| **Subgraph** | `add_node("sub", compiled_subgraph)` | 嵌套图，子图有独立状态 |

---

## 🔧 技术栈

| 组件 | 用途 |
|------|------|
| **LangGraph** | Agent 编排引擎（图、状态、条件边） |
| **LangChain** | LLM 调用抽象、工具定义 |
| **CrewAI** | 基于角色的多 Agent 框架 |
| **Qwen-Max** (阿里云百炼) | 主力 LLM（通过 OpenAI 兼容接口） |
| **Tavily** | 联网搜索工具 |
| **FastAPI** | Agent-as-a-Service HTTP 服务 |
| **SQLite** (via aiosqlite) | checkpointer 持久化存储 |
| **Pydantic** | 结构化数据校验和 Schema 定义 |

---

## 📚 推荐阅读顺序

如果你是初学者，建议按这个顺序阅读代码：

1. `tool.py` — 看看最简单的 Agent 长什么样
2. `langgraph_base.py` — **重点！** 理解图、节点、边、状态
3. `fallback.py` — 理解条件分支
4. `Interrupt _test.py` — 理解人机协作
5. `multi-router1.py` — 理解多 Agent 路由
6. `agent.workflow.py` — 理解 Plan-and-Execute
7. `research_choose.py` — 理解生产级优化

每个文件只聚焦一个概念，读的时候问自己：**"这个文件的 State 是什么？节点做了什么？边是怎么连的？"**

---

> "The best way to learn is to build. The second best way is to read someone else's build."  
> 最好的学习方式是动手构建，其次是读懂别人构建的东西。

🤖 *持续学习，持续构建 · Generated with [Claude Code](https://claude.com/claude-code)*
