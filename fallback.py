from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from config import Config
@tool
def get_precise_location_info(address: str) -> str:
    """精确位置查找（模拟，可能失败）"""
    # 模拟失败：这里故意只识别 "北京"
    if "北京" in address:
        return f"精确位置：北京市朝阳区建国门外大街1号"
    return None  # 失败
@tool
def get_general_area_info(city: str) -> str:
    """粗略地区信息"""
    return f"{city} 位于中国北部，是直辖市。"
class AgentState(dict):
    user_query: str
    primary_location_failed: bool
    location_result: str
llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1, api_key=Config.API_KEY, base_url=Config.BASE_URL)
def primary_handler(state: AgentState) -> str:
    """主处理器，尝试获取精确位置"""
    result = get_precise_location_info.invoke({"address": state["user_query"]})
    if result is None:
        return {"primary_location_failed": True, "location_result": None}
    else:
        return {"primary_location_failed": False, "location_result": result}
def fallback_handler(state: AgentState) -> str:
    if not state["primary_location_failed"]:
        return {}  # 不需要回退，结束流程
    print("🔄 回退处理器启动...")
    # 从用户查询中提取城市（简单关键词）
    city = "北京" if "北京" in state["user_query"] else "未知城市"
    result = get_general_area_info.invoke({"city": city})
    return {"location_result": result}
def response_agent(state: AgentState) -> str:
    result=state.get("location_result")
    if result:
        print(f"最终位置结果：{result}")
    else:
        print("很抱歉，无法获取位置信息。")
    return {} 
# ---------- 构建图 ----------
builder = StateGraph(AgentState)
builder.add_node("primary", primary_handler)
builder.add_node("fallback", fallback_handler)
builder.add_node("respond", response_agent)
builder.set_entry_point("primary")
builder.add_conditional_edges(
    "primary",
    lambda s: "fallback" if s.get("primary_location_failed") else "respond"
)
builder.add_edge("fallback", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# ---------- 运行 ----------
initial_state = {"user_query": "南京有什么好玩的地方"}
result = graph.invoke(initial_state)