import json
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from config import Config
@tool
def troubleshoot_issue(issue:str)->str:
     """针对技术问题提供故障排除步骤"""
     return f"故障排除步骤（针对「{issue}」）：\n1. 检查电源线是否连接。\n2. 尝试长按电源键 10 秒强制重启。\n3. 如果仍无法开机，请联系售后。"
@tool
def create_ticket(issue_type: str, details: str) -> str:
    """创建支持工单"""
    return f"工单已创建：TICKET123，问题类型：{issue_type}，详情：{details}。客服将在 24 小时内联系您。"

@tool
def escalate_to_human(issue_type: str) -> str:
    """将复杂问题转接给人工专家"""
    return f"已将「{issue_type}」问题转接给人工专家。请稍候，专家会尽快与您沟通。"
class AgentState(TypedDict):
    user_query: str
    customer_info: dict          # 包含 name, tier, recent_purchases, support_history
    support_history: Optional[List[str]]
    final_answer: str
llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1, api_key=Config.API_KEY, base_url=Config.BASE_URL)
llm_with_tools = llm.bind_tools([troubleshoot_issue, create_ticket, escalate_to_human])
#节点函数
def personalization_node(state: AgentState) -> dict:
    """个性化处理节点，根据客户信息调整响应"""
    cust=state.get("customer_info",{})
    name = cust.get("name", "valued customer")
    tier = cust.get("tier", "standard")
    purchases = cust.get("recent_purchases", [])
    history = cust.get("support_history", [])
    personalization=(
         f"\n【客户信息】\n姓名：{name}\n等级：{tier}\n"
        f"最近购买：{', '.join(purchases) if purchases else '无'}\n"
        f"历史支持记录：{'; '.join(history) if history else '无'}"
    )
     # 将个性化信息存入状态，供主节点使用
    return {"personalization_note": personalization}
def agent_node(state: AgentState):
    """主智能体节点：根据用户问题和个性化信息决策，调用工具或直接回答"""
    system_prompt = f"""您是我们电子公司的技术支持专家。

{state.get('personalization_note', '')}

对于技术问题：
1. 使用 troubleshoot_issue 工具分析问题。
2. 指导用户完成基本故障排除步骤。
3. 如果问题持续存在，使用 create_ticket 记录问题。

对于超出基本故障排除的复杂问题：
1. 使用 escalate_to_human 转接给人类专家。

保持专业但富有同理心的语气。承认技术问题可能引起的挫败感，同时提供明确的解决步骤。
"""
    messages=[
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"])
    ]
    response=llm_with_tools.invoke(messages)
    # 检查是否有工具调用
    if response.tool_calls:
         # 执行工具并将结果返回
         tool_call=response.tool_calls[0]  # 假设只调用一个工具
         tool_name=tool_call["name"]
         tool_args=tool_call["args"]
          # --- 人机交互：请求用户确认是否执行该工具 ---
         print("\n🔧 智能体建议调用工具：")
         print(f"   工具名称：{tool_name}")
         print(f"   参数：{tool_args}")
         user_input = input("\n是否执行此工具？(y/n): ").strip().lower()
    
         if user_input == 'y':
            # 用户确认，执行工具
            if tool_name == "troubleshoot_issue":
                result = troubleshoot_issue.invoke(tool_args)
            elif tool_name == "create_ticket":
                result = create_ticket.invoke(tool_args)
            elif tool_name == "escalate_to_human":
                result = escalate_to_human.invoke(tool_args)
            else:
                result = "未知工具"
            final = f"工具 {tool_name} 执行结果：\n{result}"
         else:
            # 用户拒绝，取消工具调用
            final = "您已取消工具调用。智能体将不会执行任何操作。请尝试更详细地描述问题，或手动解决。"
    else:
        # 没有工具调用，直接使用 LLM 的输出
        final = response.content
    return {"final_answer": final}
builder = StateGraph(AgentState)
builder.add_node("personalize", personalization_node)
builder.add_node("agent", agent_node)
builder.set_entry_point("personalize")
builder.add_edge("personalize", "agent")
builder.add_edge("agent", END)
graph = builder.compile()
if __name__=="__main__":
    initial_state={
        "user_query":"我的设备无法开机，已经试过重启还是不行。",
        "customer_info":{
            "name":"张三",
            "tier":"premium",
            "recent_purchases":["智能手机", "智能手表"],
            "support_history":["2023-01-15: 设备连接问题", "2023-03-22: 软件更新问题"]
        }
    }
    print("🚀 启动技术支持智能体（人机交互模式）...\n")
    result = graph.invoke(initial_state)
    print("\n" + "="*50)
    print("【最终回答】")
    print(result["final_answer"])