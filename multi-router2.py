import os
import json
import uuid
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from config import Config

# ==================== 定义工具函数（业务逻辑） ====================
def booking_handler(request: str) -> str:
    """处理航班和酒店预订的模拟函数"""
    print("-------------------------- 调用预订处理程序 ----------------------------")
    return f"已模拟对 '{request[:50]}...' 的预订操作（确认中）。"

def info_handler(request: str) -> str:
    """处理一般信息请求的模拟函数"""
    print("-------------------------- 调用信息处理程序 ----------------------------")
    return f"对 '{request[:50]}...' 的信息请求。结果：模拟信息检索完成。"

# ==================== 定义 Function Calling Schema ====================
# 预订工具的描述（用于千问模型理解何时调用）
booking_tool_schema = {
    "type": "function",
    "function": {
        "name": "booking_handler",
        "description": "处理航班和酒店的预订请求。当用户提出预订机票、酒店等相关需求时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "用户的完整预订请求内容"
                }
            },
            "required": ["request"]
        }
    }
}

# 信息查询工具的描述
info_tool_schema = {
    "type": "function",
    "function": {
        "name": "info_handler",
        "description": "回答一般性信息问题，提供知识查询、事实检索等服务。",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "用户的问题或信息请求内容"
                }
            },
            "required": ["request"]
        }
    }
}

# 工具名称到实际函数的映射
TOOL_FUNCTIONS = {
    "booking_handler": booking_handler,
    "info_handler": info_handler,
}

# ==================== 千问调用封装（OpenAI 兼容模式） ====================
from openai import OpenAI

# 初始化 OpenAI 兼容客户端（指向阿里云 DashScope）
client = OpenAI(
    api_key=Config.API_KEY,
    base_url=Config.BASE_URL,
)

def call_qwen_with_tools(
    messages: List[Dict[str, str]], 
    tools: List[Dict[str, Any]]
) -> tuple:
    """
    封装了对阿里云千问模型的带工具描述的调用请求
    调用千问模型，支持 Function Calling。
    返回 (response_message, tool_calls_required_flag)
    """
    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 让模型自动决定是否调用工具
        )
        #提取模型响应消息
        response_message = response.choices[0].message
        
        # 判断是否有工具调用请求
        tool_calls = response_message.tool_calls
        return response_message, tool_calls is not None
    except Exception as e:
        print(f"调用千问 API 出错: {e}")
        return None, False

def execute_tool_calls(response_message, tools_schemas):
    """执行模型请求的工具调用，返回工具执行结果消息"""
    if not response_message.tool_calls:
        return None
    
    tool_messages = []
    #遍历每个工具调用
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # 从映射表中找到实际函数并执行
        if function_name in TOOL_FUNCTIONS:
            #**arguments 将参数解包传递给函数。例如 booking_handler(request="巴黎酒店")
            result = TOOL_FUNCTIONS[function_name](**arguments)
            #构建 tool 角色消息
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        else:
            print(f"警告: 未找到函数 {function_name}")
    
    return tool_messages

# ==================== 定义专门 Agent（每个 Agent 封装自己的工具） ====================
class SpecialistAgent:
    """专门 Agent 基类，封装对千问模型的调用和工具执行"""
    def __init__(self, name: str, system_prompt: str, tools_schemas: List[Dict]):
        self.name = name
        self.system_prompt = system_prompt
        self.tools_schemas = tools_schemas
    
    def process(self, user_request: str) -> str:
        """处理用户请求，自动调用对应工具并返回最终答案"""
        print(f"\n--- [{self.name}] 开始处理请求: '{user_request[:60]}...' ---")
        
        # 构建消息历史
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        # 第一轮：调用模型，可能触发工具调用
        response_message, has_tool_calls = call_qwen_with_tools(messages, self.tools_schemas)
        if not response_message:
            return f"[{self.name}] 处理失败: API 调用错误"
        
        # 如果没有工具调用，直接返回模型回答
        if not has_tool_calls:
            return response_message.content
        
        # 有工具调用：将模型回复加入消息历史
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response_message.tool_calls
            ]
        })
        
        # 执行工具调用
        tool_messages = execute_tool_calls(response_message, self.tools_schemas)
        #追加结果
        if tool_messages:
            messages.extend(tool_messages)
            
            # 第二轮：将工具执行结果返回给模型，生成最终回答
            final_response, _ = call_qwen_with_tools(messages, self.tools_schemas)
            if final_response:
                return final_response.content
        
        return f"[{self.name}] 处理完成（已执行工具调用）"

# 创建专门 Agent 实例
booking_agent = SpecialistAgent(
    name="Booker",
    system_prompt=(
        "你是一个专门处理航班和酒店预订的智能体。当用户提出预订需求时，"
        "你必须调用 booking_handler 工具来完成预订操作。不要直接拒绝用户的预订请求，"
        "始终尝试通过工具来处理。"
    ),
    tools_schemas=[booking_tool_schema]
)

info_agent = SpecialistAgent(
    name="Info",
    system_prompt=(
        "你是一个专门回答一般信息问题的智能体。当用户询问任何知识类、事实类、信息类问题时，"
        "调用 info_handler 工具来获取信息。始终保持友好、有帮助的态度。"
    ),
    tools_schemas=[info_tool_schema]
)

# ==================== 协调器（核心：意图判断 + 委托） ====================
class Coordinator:
    """协调器 - 直接调用千问模型判断请求意图，然后委托给对应的专门 Agent"""
    
    def __init__(self, booking_agent: SpecialistAgent, info_agent: SpecialistAgent):
        self.booking_agent = booking_agent
        self.info_agent = info_agent
    
    def process_request(self, user_request: str) -> str:
        """
        协调器主入口：先判断意图，再委托给对应的专门 Agent
        """
        print(f"\n{'='*50}")
        print(f"🚀 协调器收到请求: '{user_request}'")
        print(f"{'='*50}")
        
        # 使用千问模型判断请求意图（不携带工具，纯分类）
        intention = self._classify_intent(user_request)
        print(f"📋 协调器判断: 意图类型 = {intention}")
        
        # 根据意图委托给专门 Agent
        if intention == "booking":
            return self.booking_agent.process(user_request)
        elif intention == "info":
            return self.info_agent.process(user_request)
        else:
            return f"协调器无法判断请求类型：'{user_request}'。请提供更明确的需求（如预订或信息咨询）。"
    
    def _classify_intent(self, user_request: str) -> str:
        """
        调用千问模型进行意图分类（booking / info / unknown）
        这里不使用 Function Calling，仅做纯文本分类
        """
        classification_prompt = f"""
        请判断以下用户请求的意图类型，只输出一个单词：
        
        用户请求: "{user_request}"
        
        规则：
        - 如果请求涉及预订航班、预订酒店、订票、行程安排等，输出 booking
        - 如果请求涉及一般信息查询、知识问答、事实查找等，输出 info
        - 如果无法判断，输出 unknown
        
        只输出一个单词，不要有其他内容。
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,  # 低温度，使分类更稳定
            )
            intention = response.choices[0].message.content.strip().lower()
            if intention in ["booking", "info", "unknown"]:
                return intention
            return "unknown"
        except Exception as e:
            print(f"意图判断失败: {e}")
            return "unknown"

# 创建协调器实例
coordinator = Coordinator(booking_agent, info_agent)

# ==================== 运行主流程 ====================
def run_coordinator(request: str) -> str:
    """运行协调器处理单个请求"""
    result = coordinator.process_request(request)
    print(f"\n✨ 最终响应: {result}\n")
    return result

def main():
    """演示协调器的路由能力"""
    print("--- 千问模型多智能体委托示例 ---")
    print("注意: 需要先设置环境变量 DASHSCOPE_API_KEY\n")
    
    # 示例请求
    test_requests = [
        ("预订", "给我在巴黎预订一家酒店。"),
        ("预订", "查找下个月去东京的航班。"),
        ("信息", "世界上最高的山是什么？"),
        ("信息", "告诉我一个随机事实。"),
    ]
    
    for desc, req in test_requests:
        print(f"\n{'*'*60}")
        print(f"测试场景 [{desc}]: {req}")
        print(f"{'*'*60}")
        result = run_coordinator(req)
        print(f"最终结果: {result[:200]}...")
        print()

if __name__ == "__main__":
    main()