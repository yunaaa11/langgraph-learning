import os
from crewai import Agent, Task, Crew,LLM
from crewai.tools import tool
import logging
from config import Config
llm = LLM(
    model=Config.LLM_MODEL,
    base_url=Config.BASE_URL,
    api_key=Config.API_KEY,
    temperature=0.1
)
## --- 1. 重构的工具：返回干净的数据 ---
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    获取给定股票代码符号的最新模拟股票价格。
    以浮点数形式返回价格。如果未找到代码，则引发 ValueError。
    """
    logging.info(f"工具调用：get_stock_price，代码为 '{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price=simulated_prices.get(ticker.upper())
    if price is not None:
        return price
    else:
        raise ValueError(f"未找到代码 '{ticker.upper()}' 的模拟价格。")

## --- 2. 定义智能体 ---
financial_analyst_agent = Agent(
    role='高级财务分析师',
    goal='使用提供的工具分析股票数据并报告关键价格。',
    backstory="你是一位经验丰富的财务分析师，擅长使用数据源查找股票信息。你提供清晰、直接的答案。",
    verbose=True,
    tools=[get_stock_price],
    # 允许委托可能很有用，但对于这个简单任务不是必需的。
    allow_delegation=False,
    llm=llm,
)
## --- 3. 精炼任务：更清晰的说明和错误处理 ---
analyze_aapl_task = Task(
    description=(
        "Apple（代码：AAPL）的当前模拟股票价格是多少？"
        "使用 'Stock Price Lookup Tool' 查找它。"
        "如果未找到代码，你必须报告无法检索价格。"
    ),
    expected_output=(
        "一个清晰的句子，说明 AAPL 的模拟股票价格。"
        "例如：'AAPL 的模拟股票价格是 $178.15。'"
        "如果无法找到价格，请明确说明。"
    ),
    agent=financial_analyst_agent,
)
## --- 4. 组建团队 ---
financial_crew = Crew(
    agents=[financial_analyst_agent],
    tasks=[analyze_aapl_task],
    verbose=True # 在生产环境中设置为 False 以获得较少的详细日志
)
def main():
    """运行团队的主函数。"""
    
    print("\n## 启动财务团队...")
    print("---------------------------------")
    
    # kickoff 方法启动执行。
    result = financial_crew.kickoff()
    
    print("\n---------------------------------")
    print("## 团队执行完成。")
    print("\n最终结果：\n", result)

if __name__ == "__main__":
    main()