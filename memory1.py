from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from config import Config
llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1, api_key=Config.API_KEY, base_url=Config.BASE_URL)
prompt=ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你是一个有用的助手。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)
## 2. 配置记忆
memory=ConversationBufferMemory(memory_key="history",return_messages=True)
## 3. 构建链
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

## 4. 运行对话
response = conversation.predict(input="嗨，我是 Jane。")
print(response)
response = conversation.predict(input="你记得我的名字吗？")
print(response)