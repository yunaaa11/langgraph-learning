from typing import Literal
from langchain_core.tools import tool
from config import Config
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
model=ChatOpenAI(model=Config.LLM_MODEL,temperature=0,
                 api_key=Config.API_KEY,
                 base_url=Config.BASE_URL)

@tool
def get_weather(city:Literal["beijing","shanghai"]):
    """获取指定城市的天气信息。"""
    if city=="beijing":
        return "It may be cloudy in beijing"
    elif city=="shanghai":
        return "It is always in shanghai"
    else:
        raise  ValueError("Unknown city")
tools=[get_weather]
#添加系统提示
# prompt="Respond in Chinese"
# prompt = """你是一个智能助手。你必须始终使用中文回答用户。
# 如果调用了工具并且工具返回英文，你必须将工具返回的内容翻译成中文后再输出。
# 不要直接输出英文。"""
#带记忆
memory=MemorySaver()
# graph=create_agent(model,tools=tools,checkpointer=memory,system_prompt=prompt)
#interrupt_before=["tools"]：指定一个节点名称列表（这里是 "tools" 节点），表示在执行这些节点之前触发中断。
graph=create_agent(model,tools=tools,interrupt_before=["tools"],checkpointer=memory)
def print_stream(stream):
    for s in stream:
        message=s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()
config={"configurable":{"thread_id":"1"}}
inputs={"messages":[("user","what is the weather in beijing?")]}
print_stream(graph.stream(inputs,config=config,stream_mode="values"))
# config={"configurable":{"thread_id":"2"}}
# inputs={"messages":[("user","what is it known for?")]}
# print_stream(graph.stream(inputs,config=config,stream_mode="values"))
#获取图的状态快照
snapshot=graph.get_state(config)
print("Next step: ",snapshot.next)
#graph.stream(None) 传入 None 作为输入，表示“继续执行之前中断的图”。
print_stream(graph.stream(None,config=config,stream_mode="values"))
graph_png=graph.get_graph().draw_mermaid_png()
with open("react_case.png","wb") as f:
    f.write(graph_png)

