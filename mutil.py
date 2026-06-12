import os
from crewai import Agent, Task, Crew,LLM, Process
from crewai.tools import tool
import logging
from config import Config
llm = LLM(
    model=Config.LLM_MODEL,
    base_url=Config.BASE_URL,
    api_key=Config.API_KEY,
    temperature=0.1
)
# 定义具有特定角色和目标的 Agent
researcher = Agent(
        role='高级研究分析师',
        goal='查找并总结 AI 的最新趋势。',
        backstory="你是一位经验丰富的研究分析师，擅长识别关键趋势和综合信息。",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
writer = Agent(
    role='技术内容作家',
        goal='基于研究发现撰写清晰且引人入胜的博客文章。',
        backstory="你是一位熟练的作家，可以将复杂的技术主题转化为易于理解的内容。",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
# 为智能体定义任务
research_task = Task(
        description="研究 2024-2025 年人工智能中出现的前 3 个趋势。重点关注实际应用和潜在影响。",
        expected_output="前 3 个 AI 趋势的详细摘要，包括关键点和来源。",
        agent=researcher,
    )

writing_task = Task(
        description="基于研究发现撰写一篇 500 字的博客文章。文章应该引人入胜且易于普通读者理解。",
        expected_output="一篇关于最新 AI 趋势的完整 500 字博客文章。",
        agent=writer,
        context=[research_task],
    )
blog_creation_crew=Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    llm=llm,
    verbose=True,  # 使用布尔值，开启详细日志
)
print("##运行博客创建团队... ##")
try:
        result = blog_creation_crew.kickoff()
        print("\n------------------\n")
        print("## 团队最终输出 ##")
        print(result)
except Exception as e:
        print(f"\n发生意外错误：{e}")
