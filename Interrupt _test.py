from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
class State(TypedDict):
    input:str
def step_1(state):
    print("---Step 1---")
    pass
def step_2(state):
    print("---Step 2---")
    pass
def step_3(state):
    print("---Step 3---")
    pass
builder=StateGraph(State)
builder.add_node("step_1",step_1)
builder.add_node("step_2",step_2)
builder.add_node("step_3",step_3)
builder.add_edge(START,"step_1")
builder.add_edge("step_1","step_2")
builder.add_edge("step_2","step_3")
builder.add_edge("step_3",END)
memory=MemorySaver()
#当准备进入 step_3 时，它会将当前所有的状态（State）保存到 MemorySaver 中，然后挂起任务。
graph=builder.compile(checkpointer=memory,interrupt_before=["step_3"])
graph_png=graph.get_graph().draw_mermaid_png()
with open("langgraph_case.png","wb") as f:
    f.write(graph_png)
initial_input={"input":"hello world"}
thread={"configurable":{"thread_id":"1"}}
#第一次流式运行 (运行到中断)
for event in graph.stream(initial_input,thread,stream_mode="values"):
    print(event)
#人工审批逻辑
user_approval=input("Do you want to go to Step 3?(yes/no)")
#恢复运行
if user_approval.lower()=="yes":
#None现在不需要新数据，直接从上次停下的地方（step_3 之前）继续跑就行了
    for event in graph.stream(None,thread,stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")