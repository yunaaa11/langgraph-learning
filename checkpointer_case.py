import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
async def main():
    #创建一个StateGraph对象，节点值类型int
    builder=StateGraph(int)
    builder.add_node("add_one",lambda x:x+1)
    builder.set_entry_point("add_one")
    builder.set_finish_point("add_one")
    #AsyncSqliteSaver 作为检查点保存器（checkpointer），它会把图运行过程中的每一步状态都存到本地的 SQLite 文件 checkpoints.db 中
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        #便于状态图，使用memory作为检查点保存器
        graph=builder.compile(checkpointer=memory)
        #创建一个异步调用的协程，输入值为1，并传入而额外的配置参数
        result=await graph.ainvoke(1,{"configurable":{"thread_id":"thread-1"}})
        print(result)
if __name__ == "__main__":
    asyncio.run(main())