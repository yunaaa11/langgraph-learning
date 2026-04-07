import os
from dotenv import load_dotenv

load_dotenv() 
class Config:
      API_KEY=os.getenv("API_KEY")
      BASE_URL=os.getenv("BASE_URL")
      LLM_MODEL=os.getenv("LLM_MODEL")
      TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")