import operator
from typing import Annotated, List, TypedDict, Union
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. 定义状态 (State)
# ==========================================

# 全局父图的状态
class OverallState(TypedDict):
    logs: str  # 输入的原始日志
    docs: str  # 转换后的文档
    # 使用 Annotated[list, operator.add] 来聚合不同并行支路的报告
    final_reports: Annotated[List[str], operator.add]

# 子图 1 (Failure Analysis) 的状态
class FailureAnalysisState(TypedDict):
    docs: str          # 从父图传入的文档
    failures: List[str] # 本地私有变量：故障列表
    summary: str       # 本地私有变量：故障总结

# 子图 2 (Question Summarization) 的状态
class QuestionSummaryState(TypedDict):
    docs: str          # 从父图传入的文档
    raw_summary: str   # 本地私有变量：原始总结
    formatted_report: str # 本地私有变量：格式化后的 Slack 报告


# ==========================================
# 2. 定义节点函数 (Node Functions)
# ==========================================

# --- 父图公共节点 ---
def convert_logs_to_docs(state: OverallState):
    print("--- [父图] 运行: convert_logs_to_docs ---")
    return {"docs": f"processed_docs_from_{state['logs']}"}

# --- 子图 1 (Failure Analysis) 节点 ---
def get_failures(state: FailureAnalysisState):
    print("--- [子图1] 运行: get_failures ---")
    return {"failures": ["failure_A", "failure_B"]}

def failure_generate_summary(state: FailureAnalysisState):
    print("--- [子图1] 运行: generate_summary (Failure) ---")
    # 结合传入的文档和本地的故障列表进行总结
    return {"summary": f"Summary of {len(state['failures'])} failures found in {state['docs']}"}

# --- 子图 2 (Question Summarization) 节点 ---
def generate_raw_summary(state: QuestionSummaryState):
    print("--- [子图2] 运行: generate_summary (Questions) ---")
    return {"raw_summary": "raw_summary_of_questions"}

def send_to_slack(state: QuestionSummaryState):
    print("--- [子图2] 运行: send_to_slack (Simulation) ---")
    # 这里模拟一个发送动作，不需要返回状态更新
    return {}

def format_report_for_slack(state: QuestionSummaryState):
    print("--- [子图2] 运行: format_report_for_slack ---")
    return {"formatted_report": f"Slack Report: {state['raw_summary']}"}


# ==========================================
# 3. 构建子图 (Build Subgraphs)
# ==========================================

# --- 构建子图 1: Failure Analysis ---
failure_analysis_builder = StateGraph(FailureAnalysisState)
failure_analysis_builder.add_node("get_failures", get_failures)
# 注意：为了避免节点名冲突，我们在子图内部起具体名字，但图片中显示为 generate_summary
failure_analysis_builder.add_node("generate_summary_f", failure_generate_summary)

failure_analysis_builder.add_edge(START, "get_failures")
failure_analysis_builder.add_edge("get_failures", "generate_summary_f")
# 子图内部只需要连到 generate_summary_f，父图调用时会自动映射到子图的 __end__
failure_analysis_builder.add_edge("generate_summary_f", END)
failure_analysis = failure_analysis_builder.compile()

# --- 构建子图 2: Question Summarization ---
question_summarization_builder = StateGraph(QuestionSummaryState)
question_summarization_builder.add_node("generate_summary_q", generate_raw_summary)
question_summarization_builder.add_node("send_to_slack", send_to_slack)
question_summarization_builder.add_node("format_report_for_slack", format_report_for_slack)

question_summarization_builder.add_edge(START, "generate_summary_q")
question_summarization_builder.add_edge("generate_summary_q", "send_to_slack")
question_summarization_builder.add_edge("send_to_slack", "format_report_for_slack")
question_summarization_builder.add_edge("format_report_for_slack", END)
question_summarization = question_summarization_builder.compile()


# ==========================================
# 4. 构建并连接父图 (Build Parent Graph)
# ==========================================

# --- 处理子图返回值的 Reducer 函数 ---
# 由于子图的状态类和父图不同，我们需要手动将子图的最终结果映射回父图的状态。
# LangGraph 在处理编译好的子图节点时支持这种显式映射。

def fa_subgraph_adapter(sub_state: FailureAnalysisState):
    """适配子图1的输出到父图"""
    # 将子图的本地 summary 添加到父图的 final_reports 列表中
    return {"final_reports": [f"[支路1] {sub_state['summary']}"]}

def qs_subgraph_adapter(sub_state: QuestionSummaryState):
    """适配子图2的输出到父图"""
    return {"final_reports": [f"[支路2] {sub_state['formatted_report']}"]}


builder = StateGraph(OverallState)

# 添加普通节点
builder.add_node("convert_logs_to_docs", convert_logs_to_docs)

# 核心修改：将编译好的子图作为一个节点添加到父图中，并指定适配器
builder.add_node("failure_analysis", failure_analysis)
builder.add_node("question_summarization", question_summarization)

# --- 5. 建立连接 (实现并行与合并) ---
builder.add_edge(START, "convert_logs_to_docs")

# 从同一节点发出多条边，实现并行执行
builder.add_edge("convert_logs_to_docs", "failure_analysis")
builder.add_edge("convert_logs_to_docs", "question_summarization")

# 并行支路会自动在 __end__ 节点合并
builder.add_edge("failure_analysis", END)
builder.add_edge("question_summarization", END)

graph = builder.compile()

# ==========================================
# 6. 生成图表
# ==========================================
graph_png=graph.get_graph(xray=True).draw_mermaid_png()
with open("langgraph_sub.png","wb") as f:
    f.write(graph_png)


# ==========================================
# 7. 运行图实例
# ==========================================
print("\n\n--- 开始运行复杂工作流 ---")
initial_input = {"logs": "server_log_2023_10_27.txt", "final_reports": []}

# 使用 stream 查看更新过程
for chunk in graph.stream(initial_input, version="v2"):
    if chunk["type"] == "updates":
        print(f"状态更新: {chunk['data']}")

# 获取最终结果
final_state = graph.get_state({"logs": "server_log_2023_10_27.txt"})
print("\n\n=== 最终合并报告 ===")
for report in final_state.values.get("final_reports", []):
    print(report)
print("====================")