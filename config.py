import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


class Config:
    # ── Groq（主力模型）──
    API_KEY   = os.getenv("API_KEY")
    BASE_URL  = os.getenv("BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")

    # ── Tavily（备用搜索，需自行注册）──
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
