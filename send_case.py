import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Send
from langgraph.graph import END,START
class OverallState(TypedDict):
    subject:list[str]
    jokes:Annotated[list[str],operator.add]

def continue_to_jokes(state:OverallState)-> Literal["generate_joke"]:
    #返回一个send对象的列表，每个对象包含一个"generate_joke"的命令和对应主题的字典
    return [Send("generate_joke",{"subject":s}) for s in state["subject"]]

builder=StateGraph(OverallState)
builder.add_node("generate_joke",lambda state:{"jokes":[f"Joke about {state['subject']}"]})
builder.add_conditional_edges(START,continue_to_jokes)
# builder.add_conditional_edges(START, continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke",END)
graph=builder.compile()

result=graph.invoke({"subject":["cats","dogs"]})
print(result)
graph_png=graph.get_graph().draw_mermaid_png()
with open("send_case.png","wb") as f:
    f.write(graph_png)
