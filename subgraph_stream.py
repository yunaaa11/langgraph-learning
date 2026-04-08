import asyncio
import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # 1. 导入存档工具

# --- 状态定义 (保持不变) ---
class OverallState(TypedDict):
    logs: str
    docs: str
    final_reports: Annotated[List[str], operator.add]

class FailureAnalysisState(TypedDict):
    docs: str
    summary: str

class QuestionSummaryState(TypedDict):
    docs: str
    formatted_report: str

# --- 节点函数 (保持不变) ---
async def convert_logs_to_docs(state: OverallState):
    await asyncio.sleep(0.5)
    return {"docs": f"DOCS_DATA_OF_{state['logs']}"}

# 子图逻辑
async def failure_analysis_logic(state: FailureAnalysisState):
    await asyncio.sleep(1)
    return {"summary": "检测到 2 处网络连接超时异常。"}

async def question_summary_logic(state: QuestionSummaryState):
    await asyncio.sleep(0.8)
    return {"formatted_report": "用户提问主要集中在 API 鉴权失败问题。"}

# --- 子图构建 ---
fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("analysis_node", failure_analysis_logic)
fa_builder.add_edge(START, "analysis_node")
fa_builder.add_edge("analysis_node", END)
fa_graph = fa_builder.compile()

qs_builder = StateGraph(QuestionSummaryState)
qs_builder.add_node("summary_node", question_summary_logic)
qs_builder.add_edge(START, "summary_node")
qs_builder.add_edge("summary_node", END)
qs_graph = qs_builder.compile()

# --- 父图构建 ---
#state["xxx"]：是从全局档案库里取东西
async def call_fa_subgraph(state: OverallState):
    res = await fa_graph.ainvoke({"docs": state["docs"]})
    #operator.add，所以 LangGraph 不会把旧的删掉，而是把这个新列表加进旧列表里。
    return {"final_reports": [f"【故障分析】: {res['summary']}"]}

async def call_qs_subgraph(state: OverallState):
    res = await qs_graph.ainvoke({"docs": state["docs"]})
    return {"final_reports": [f"【问答统计】: {res['formatted_report']}"]}

builder = StateGraph(OverallState)
builder.add_node("convert_logs_to_docs", convert_logs_to_docs)
builder.add_node("fa_branch", call_fa_subgraph)
builder.add_node("qs_branch", call_qs_subgraph)
builder.add_edge(START, "convert_logs_to_docs")
builder.add_edge("convert_logs_to_docs", "fa_branch")
builder.add_edge("convert_logs_to_docs", "qs_branch")
builder.add_edge("fa_branch", END)
builder.add_edge("qs_branch", END)

# 2. 核心修改：添加 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph_png=graph.get_graph(xray=True).draw_mermaid_png()
with open("streaming_update.png","wb") as f:
    f.write(graph_png)

# --- 运行流式输出 ---
async def main():
    inputs = {"logs": "server_v2.log", "final_reports": []}
    # 3. 核心修改：运行时必须提供 config 指定 thread_id
    config = {"configurable": {"thread_id": "test_run_1"}}
    
    print("🚀 [开始运行] 正在启动异步流式更新...")
    print("-" * 50)
    
    # # 传入 config   Updates 模式
    # async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
    #     for node_name, updated_values in chunk.items():
    #         print(f"节点: {node_name}, 更新了: {updated_values}")
    #         print("-" * 30)
    # --- 修改为 Values 模式 ---
    async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
    # 在 values 模式下，chunk 直接就是当前的整个 State 字典
      print(f"当前完整状态: {chunk}")

    # 4. 再次获取状态时也传入 config，这样它就能找到对应的存档了
    final_state = await graph.aget_state(config)
    print("\n✅ [运行结束] 最终合并的报告列表：")
    for r in final_state.values.get("final_reports", []):
        print(f" * {r}")

if __name__ == "__main__":
    asyncio.run(main())