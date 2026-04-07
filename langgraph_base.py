from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END,StateGraph,MessagesState
from langgraph.prebuilt import ToolNode
from config import Config 
#定义工具函数，用于代理调用外部工具
@tool
def search(query:str):
    """模拟一个搜索工具"""
    if "上海" in query.lower() or "Shanghai" in query.lower():
        return "现在30度，有雾"
    return "现在35度，晴天"
#将工具函数放入工具列表
tools=[search]
#创建工具节点
tool_node=ToolNode(tools)
#1.初始化模型和工具，定义并般的工具到模型
model=ChatOpenAI(model=Config.LLM_MODEL,temperature=0,
                 api_key=Config.API_KEY,
                 base_url=Config.BASE_URL).bind_tools(tools)

#定义函数，决定是否继续执行
def should_continue(state:MessagesState)->Literal["tools","__end__"]:
#Literal["tools",END] 定义两条边 一个工具一个节点
    messages=state['messages']#h获取目前所有的对话历史
    last_message=messages[-1]#获取对话历史中的最后一条消息
    #如果llm调用工具，则转到"tools"节点
    if last_message.tool_calls:
        return "tools"
    #否则，停止（回复用户)
    return END

#定义调用模型的函数
def call_model(state:MessagesState):
    messages=state['messages']
    response=model.invoke(messages)#把整个对话历史发给大模型
    #返回列表，因为将被添加到现有列表中
    return {"messages":[response]}#追加（Append） 到旧消息列表的末尾

#2.用状态初始化图，定义一个新的状态图
workflow=StateGraph(MessagesState)

#3.定义图节点，定义将循环的两个节点
workflow.add_node("agent",call_model)
workflow.add_node("tools",tool_node)

#4.定义入口点和图边
#设置入口点为"agent" 这意味这这是第一个被调用的节点
workflow.set_entry_point("agent")
#添加条件边
workflow.add_conditional_edges(
    #定义起始节点agent,这些边在调用agnet节点后采纳
    "agent",
    #接下来，传递决定下一个调用节点的函数
    should_continue,#“逻辑判断员”：它只负责输出一个结果（比如 "tools" 或 END）。
    {
        "tools": "tools",  # 如果函数返回 "tools"，去往 tools 节点
        END: END           # 如果函数返回 END，去往结束标记
    }
)
#添加从"tools"到"agent"的普通边
#在调用"tools"后，接下来到"agent"节点
workflow.add_edge("tools",'agent')
#初始化内存以在图运行之间持久化状态
checkpointer=MemorySaver()

#5.编译图
#编译成一个langchain可运行对象 可以像使用其他可运行对象引用使用它
#我们（可选地）在编译图时传递内容
app=workflow.compile(checkpointer=checkpointer)

#6.执行图,使用可运行对象
final_state=app.invoke(
    {"messages":[HumanMessage(content="上海天气怎么样?")]},
    config={"configurable":{"thread_id":42}}
)
result=final_state["messages"][-1].content
print(result)
final_state=app.invoke(
    {"messages":[HumanMessage(content="我问的哪个城市")]},
    config={"configurable":{"thread_id":41}}
)
#从final_state中获取最后一条信息的内容
result=final_state["messages"][-1].content
print(result)
#将生成的图片保存到文件
graph_png=app.get_graph().draw_mermaid_png()
with open("langgraph_base.png","wb") as f:
    f.write(graph_png)






