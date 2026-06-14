import asyncio
import os
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from config import Config
from dotenv import load_dotenv

load_dotenv()
class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str] = []
    examples: List[str] = []

class AgentCapabilities(BaseModel):
    streaming: bool = False
class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    capabilities: AgentCapabilities = AgentCapabilities()
    skills: List[AgentSkill] = []
@tool
def check_availability(original_query: str) -> str:
    """
    检查用户日历在指定时间范围内是否空闲。
    参数应为自然语言时间段，例如 "tomorrow 10am to 11am"。
    返回空闲状态或忙碌详情。
    """
    # 模拟实现：简单判断关键词
    if "busy" in original_query.lower():
        return "检测到忙碌，有会议安排。"
    elif "free" in original_query.lower() or "10am to 11am" in original_query:
        return "您在该时间段内是空闲的。"
    else:
        # 尝试提取时间段
        return "正在查询... (模拟结果: 空闲)"
llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1, api_key=Config.API_KEY, base_url=Config.BASE_URL)
tools = [check_availability]
prompt = ChatPromptTemplate.from_messages([
    ("system", 
 "你是一个日历助手。对于任何用户问题，你**必须**调用 check_availability 工具，"
 "并且将用户的**完整原始消息**（一字不改）作为参数 original_query 传入。"
 "禁止自己提取或修改用户的消息内容。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
agent=create_tool_calling_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

app = FastAPI(title="LangChain A2A Server")
# 定义 AgentCard（静态）
agent_card = AgentCard(
    name="Calendar Agent (LangChain)",
    description="An agent that can check user availability using Google Calendar",
    url="http://localhost:8000/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="check_availability",
            name="Check Availability",
            description="Checks a user's availability for a time range",
            tags=["calendar"],
            examples=["Am I free from 10am to 11am tomorrow?"]
        )
    ]
)
# 存储任务结果（简单内存存储，生产环境可用 Redis 或数据库）
tasks_store: Dict[str, Dict[str, Any]] = {}
@app.get("/.well-known/agent-card")
async def get_agent_card():
    """A2A 标准端点：返回 AgentCard JSON"""
    return agent_card.dict()

@app.post("/tasks")
async def create_task(background_tasks: BackgroundTasks, request: dict):
    """
    A2A 标准端点：创建新任务。
    请求体应包含 skill_id 和 input。
    """
    skill_id=request.get("skill_id")
    user_input=request.get("input")
    if not skill_id or not user_input:
        raise HTTPException(status_code=400, detail="Missing skill_id or input")
    task_id = str(uuid.uuid4())
    tasks_store[task_id] = {"status": "pending", "result": None}#在内存中存储所有任务的状态
    background_tasks.add_task(run_agent_task,task_id,user_input)
    return {"task_id":task_id,"status":"pending"}
async def run_agent_task(task_id:str,user_input:str):
    """后台运行 LangChain 智能体，更新任务存储"""
    print(f"后台任务启动，task_id={task_id}, input={user_input}")
    try:
       result=agent_executor.invoke({"input":user_input})
       tasks_store[task_id]={"status":"completed","result":result}
    except Exception as e:
        tasks_store[task_id]={"status":"failed","result":str(e)}
@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """A2A 标准端点：查询任务状态和结果"""
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
# ---------- 7. 新增端点：同步请求（sendTask） ----------
@app.post("/sendTask")
async def send_task(request: dict):
    """
    同步请求：直接执行智能体，等待完整结果后返回。
    请求体应包含 message.parts[0].text 等（简化版，只取 input 字段）。
    """
    # 从 A2A 标准格式中提取用户输入（简化处理）
    user_input = None
    if "message" in request and "parts" in request["message"]:
        for part in request["message"]["parts"]:
            if part.get("type") == "text":
                user_input = part.get("text")
                break
    if not user_input:
        # 兼容简单格式
        user_input = request.get("input")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # 直接同步执行（等待结果）
    try:
        result = await agent_executor.ainvoke({"input": user_input})
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "status": "completed",
                "output": result["output"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 8. 新增端点：流式请求（sendTaskSubscribe）使用 SSE ----------
@app.post("/sendTaskSubscribe")
async def send_task_subscribe(request: dict):
    """
    流式请求：建立 SSE 连接，逐步返回智能体的部分输出。
    """
    user_input = None
    if "message" in request and "parts" in request["message"]:
        for part in request["message"]["parts"]:
            if part.get("type") == "text":
                user_input = part.get("text")
                break
    if not user_input:
        user_input = request.get("input")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'start', 'task_id': str(uuid.uuid4())})}\n\n"
        try:
            async for event in agent_executor.astream_events(
                {"input": user_input},
                version="v1"
            ):
                event_type = event["event"]
                if event_type == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
                elif event_type == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
                elif event_type == "on_tool_end":
                    # 工具的返回值在 event["data"]["output"] 中
                    output = event["data"].get("output")
                    if output:
                        yield f"data: {json.dumps({'type': 'tool_end', 'output': output})}\n\n"
            # 完成事件
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
if __name__ == "__main__":
    import uvicorn
    import json 
    uvicorn.run(app, host="0.0.0.0", port=8000)
