import json
import sys
import io
import requests
from openai import OpenAI
from config import Config

# 修复 Windows GBK 编码下 print emoji/特殊字符 报错的问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 客户端 ──
client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

# ── Groq 免费模型（根据你的 API Key 实际可用的模型选择）──
MODEL_SIMPLE    = "meta-llama/llama-4-scout-17b-16e-instruct"  # 17B · 快
MODEL_REASONING = "llama-3.3-70b-versatile"                   # 70B · 推理最强
MODEL_SEARCH    = "openai/gpt-oss-120b"                       # 120B · 综合最强


## ── 步骤 1：分类 ──
def classify_prompt(prompt: str) -> dict:
    """让 LLM 判断问题类型：simple / reasoning / internet_search"""
    system_message = {
        "role": "system",
        "content": (
            "You are a classifier that analyzes user prompts and returns one of three categories ONLY:\n\n"
            "- simple\n"
            "- reasoning\n"
            "- internet_search\n\n"
            "Rules:\n"
            "- Use 'simple' for direct factual questions that need no reasoning or current events.\n"
            "- Use 'reasoning' for logic, math, or multi-step inference questions.\n"
            "- Use 'internet_search' if the prompt refers to current events, recent data, "
            "or things not in your training data.\n\n"
            "Respond ONLY with JSON like:\n"
            '{ "classification": "simple" }'
        ),
    }
    user_message = {"role": "user", "content": prompt}
    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[system_message, user_message],
        temperature=0,
    )
    reply = response.choices[0].message.content
    return json.loads(reply)


## ── 步骤 2：Tavily 搜索──
def tavily_search(query: str, num_results: int = 3) -> list:
    """用 Tavily Search 搜索，返回 [title, snippet, link] 列表"""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": Config.TAVILY_API_KEY,
                "query": query,
                "max_results": num_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("content", "")[:500],
                "link":    r.get("url", ""),
            }
            for r in data.get("results", [])
        ]
    except Exception as e:
        return [{"title": "搜索出错", "snippet": str(e), "link": ""}]


## ── 步骤 3：生成响应 ──
def generate_response(prompt: str, classification: str, search_results=None) -> tuple:
    """返回 (回答文本, 使用的模型名)"""
    if classification == "simple":
        model = MODEL_SIMPLE
        full_prompt = prompt

    elif classification == "reasoning":
        model = MODEL_REASONING
        full_prompt = prompt

    elif classification == "internet_search":
        model = MODEL_SEARCH
        if search_results:
            context_lines = [
                f"- {r['title']}: {r['snippet']} ({r['link']})"
                for r in search_results
            ]
            search_context = "\n".join(context_lines)
        else:
            search_context = "未找到搜索结果。"
        full_prompt = (
            f"请用中文，基于以下搜索结果回答用户问题。\n\n"
            f"搜索结果：\n{search_context}\n\n"
            f"用户问题：{prompt}"
        )
    else:
        model = MODEL_SIMPLE
        full_prompt = prompt

    # ⬇ API 统一调用
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content, model


## ── 步骤 4：路由器 ──
def handle_prompt(prompt: str) -> dict:
    """根据分类自动路由：简单回答 / 推理 / 联网搜索"""
    classification_result = classify_prompt(prompt)
    classification = classification_result["classification"]

    search_results = None
    if classification == "internet_search":
        search_results = tavily_search(prompt)

    answer, model = generate_response(prompt, classification, search_results)
    return {
        "classification": classification,
        "response":       answer,
        "model":          model,
    }


if __name__ == "__main__":
    #test_prompt = "What is the capital of Australia?"
    # test_prompt = "Explain the impact of quantum computing on cryptography."
    test_prompt = "What are the latest developments in AI in 2026?"

    result = handle_prompt(test_prompt)
    print("🔍 Classification:", result["classification"])
    print("🧠 Model Used:", result["model"])
    print("🧠 Response:\n", result["response"])
