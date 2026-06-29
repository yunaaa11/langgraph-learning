from dotenv import load_dotenv
from config import Config
from langchain_tavily import TavilySearch
tools=[TavilySearch(max_results=3)]  # 增加搜索结果数，确保能查到准确信息
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.prebuilt import create_react_agent
load_dotenv()

# 系统提示：必须明确指示模型使用搜索工具
system_prompt = (
    "你是一个搜索助手。对于用户提出的任何问题，你必须使用 TavilySearch 工具进行搜索，"
    "然后基于搜索结果用中文给出准确回答。不要凭自己的知识猜测，一定要先搜索再回答。"
)

llm=ChatOpenAI(model=Config.LLM_MODEL,
                 api_key=Config.API_KEY,
                 base_url=Config.BASE_URL)
agent_executor=create_react_agent(llm,tools,prompt=system_prompt)
#agent_executor.invoke({"messages":[{"user","谁是美国公开赛的获胜者"}]})
import operator
from typing import Annotated,List,Tuple,TypedDict
#定义TypeDict类PlanExecute,用于存储输入、计划、过去的步骤和响应
class PlanExecute(TypedDict):
    input:str
    plan:List[str]
    past_steps:Annotated[List[Tuple],operator.add]
    response:str
from pydantic import BaseModel,Field
#定义一个Plan模型类，用于描述未来要执行的计划
class Plan(BaseModel):
    steps:List[str]=Field(
        description="需要执行的不同步骤，应该按顺序排列"
    )
from langchain_core.prompts import ChatPromptTemplate
#创建一个计划生成的提示词模板
planner_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "对于给定的目标，提出一个简单的逐步计划，这个计划一个包含独立的任务，如果正确执行将得出正确的答案，不要添加任何多余的步骤。最后一步的结果一个最终答案，确保每一步都有所有必要的信息-不要跳过步骤。\n\n请严格按照以下JSON格式输出，不要输出其他内容：\n{{\"steps\": [\"步骤1\", \"步骤2\", \"步骤3\"]}}"
        ),
        ("placeholder","{messages}"),
    ]
)
#使用指定的提示模板创建一个计划生成器
planner=planner_prompt|ChatOpenAI(
    model=Config.LLM_MODEL,
    temperature=0,
    api_key=Config.API_KEY,
    base_url=Config.BASE_URL
).with_structured_output(Plan)
from typing import Union
class Response(BaseModel):
    response:str
#类中属性action，类型Union[Response,Plan],表示二选一，回复还是做计划
class Act(BaseModel):
    action:Union[Response,Plan]=Field(
        description="要执行的行为。如果要会有用户，使用Response。如果需要进一步使用工具获取答案，使用Plan"
    )
#创建一个重新计划的提示模板
replanner_prompt=ChatPromptTemplate.from_template(
    """对于给定的目标，提出一个简单的逐步计划。这个计划一个包含独立的任务，如果正确执行将得到正确的答案。不要添加任何多余的步骤。最后一步的结果一个最终答案，确保每一步都有所有必要的信息-不要跳过步骤
    你的目标是：
    {input}
    你的原计划是：
    {plan}
    你目前已经完成的步骤是：
    {past_steps}
    相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤，不要返回已经完成的步骤作为计划的一部分。

    请以JSON格式输出：
    - 如果需要继续执行步骤，请输出JSON：{{"action": {{"steps": ["剩余步骤1", "剩余步骤2"]}}}}
    - 如果可以回复用户了，请输出JSON：{{"action": {{"response": "你的最终回答"}}}}"""
)
#使用指定的提示词模板创建一个重新计划生成器
replanner=replanner_prompt|ChatOpenAI(
     model=Config.LLM_MODEL,
    temperature=0,
    api_key=Config.API_KEY,
    base_url=Config.BASE_URL
).with_structured_output(Act)
from typing import Literal
   #定义一个异步函数，用于生成计划步骤
async def plan_step(state:PlanExecute):
        plan=await planner.ainvoke({"messages":[("user",state["input"])]})
        return {"plan":plan.steps}
    #用于执行步骤
async def execute_step(state:PlanExecute):
        plan=state["plan"]
        plan_str="\n".join(f"{i+1}.{step}"for i,step in enumerate(plan))
        task=plan[0]                                                    # 取当前计划的第一个步骤
        completed_count=len(state.get("past_steps",[]))                 # 已完成的步骤数
        step_num=completed_count+1                                      # 当前是第几步
        task_formatted=f"""对于以下计划：
    {plan_str}\n\n你的任务是执行第{step_num}步，{task}。请务必使用搜索工具查找最新信息。"""
        agent_response=await agent_executor.ainvoke(
            {"messages":[("user",task_formatted)]}
        )
        return {
            "past_steps":state["past_steps"]+[(task,agent_response["messages"][-1].content)],
            "plan":plan[1:],  # 移除已完成的步骤，只保留剩余步骤
        }
    #用于重新计划步骤
async def replan_step(state:PlanExecute):
        output=await replanner.ainvoke(state)
        if isinstance(output.action,Response):
            return {"response":output.action.response}
        else:
            return {"plan":output.action.steps}
    #用于判断是否结束
def should_end(state:PlanExecute)->Literal["agent","__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
from langgraph.graph import START, StateGraph
#创建一个状态图，初始化PlanExecute
workflow=StateGraph(PlanExecute)
#添加计划节点
workflow.add_node("plan",plan_step)
#执行步骤节点
workflow.add_node("agent",execute_step)
#重新计划节点
workflow.add_node("replan",replan_step)
#开始到计划节点的边
workflow.add_edge(START,"plan")
#计划到代理节点的边
workflow.add_edge("plan","agent")
#代理到重新计划节点的边
workflow.add_edge("agent","replan")
#添加条件边，用于下一步操作者
workflow.add_conditional_edges(
    "replan",
    #传入判断函数，确定下一个节点
    should_end,
    {
        "agent": "agent", 
        "__end__": "__end__"
    }
)
#编译状态图，生成langchain可运行对象
app=workflow.compile()
#生成的图片保存到文件
graph_png=app.get_graph().draw_mermaid_png()
with open("agent_workflow.png", "wb") as f:
    f.write(graph_png)
#设置配置，递归限制50
config={"recursion_limit":50}
#输入数据
inputs={"input":"2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里？请用中文回答"}
# --- 6. 运行主函数 ---
async def main():
    # 保存架构图
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open("agent_workflow.png", "wb") as f: # 必须使用 wb 模式
            f.write(graph_png)
    except:
        print("无法生成图片，请检查是否安装了 pyppeteer 或其他绘图依赖")

    inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里？请用中文回答", "past_steps": []}
    config = {"recursion_limit": 50}

    # 异步流式输出
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(f"\n[Node: {k}]")
                print(v)

if __name__ == "__main__":
    asyncio.run(main())