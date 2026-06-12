import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from config import Config
from langchain_openai import ChatOpenAI
try:
    llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1, api_key=Config.API_KEY, base_url=Config.BASE_URL)
    print(f"语言模型已初始化：{llm.model}")
except Exception as e:
    print(f"初始化语言模型时出错：{e}")
    llm = None
@langchain_tool
def search_information(query: str) -> str:
    """提供有关给定主题的事实信息。使用此工具查找诸如"法国首都"或"伦敦的天气？"等短语的答案。
    """
    print(f"\n--- 🛠️ 工具调用：search_information，查询：'{query}' ---")
    
    # 使用预定义结果字典模拟搜索工具。
    simulated_results = {
        "伦敦的天气": "伦敦目前多云，温度为 15°C。",
    "法国的首都": "法国的首都是巴黎。",
    "地球人口": "地球的估计人口约为 80 亿人。",
    "最高的山": "珠穆朗玛峰是海拔最高的山峰。",
    "default": f"'{query}' 的模拟搜索结果：未找到特定信息..."
    }
    
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- 工具结果：{result} ---")
    return result
tools=[search_information]
if llm:
    agent_prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent=create_tool_calling_agent(llm, tools, agent_prompt)
    # AgentExecutor 是调用智能体并执行所选工具的运行
    agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)
async def run_agent_with_tools(query:str):
    """使用查询调用智能体执行器并打印最终响应。"""
    print(f"\n--- 🏃 使用查询运行智能体：'{query}' ---")
    try:
        response=await agent_executor.ainvoke({"input":query})
        print("\n--- ✅ 最终智能体响应 ---")
        print(response["output"])
    except Exception as e:
        print(f"\n🛑 智能体执行期间发生错误：{e}")
async def main():
    """并发运行所有智能体查询。"""
    tasks=[
        run_agent_with_tools("法国的首都是哪里？"),
        run_agent_with_tools("伦敦的天气？"),
        run_agent_with_tools("告诉我一些关于狗的事情"),
    ]
    #并发执行多个协程（tasks 列表中的每个元素都是一个协程对象），等待所有协程执行完毕
    #*tasks 将列表解包成多个参数，等价于 asyncio.gather(task1, task2, task3)
    #协程会同时开始运行
    await asyncio.gather(*tasks)
nest_asyncio.apply()
asyncio.run(main())
