from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from agent import AgentState, call_llm, should_continue, take_action


memory = MemorySaver()

def builder() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)

    graph.add_edge(START, "llm")
    graph.add_edge("retriever_agent", "llm")

    graph.add_conditional_edges(
        source="llm",
        path=should_continue,
        path_map={True: "retriever_agent", False: END},
    )
    return graph.compile(checkpointer=memory)
