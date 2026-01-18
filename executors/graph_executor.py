# executors/graph_executor.py (future)

from langgraph.graph import StateGraph, END

def build_retry_graph(tool, max_retries=3):
    graph = StateGraph(AgentState)
    
    graph.add_node("execute", execute_tool)
    graph.add_node("validate", validate_result)
    graph.add_node("fix", ask_llm_to_fix)
    
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: "end" if state["valid"] or state["retries"] >= max_retries else "fix",
        {"end": END, "fix": "fix"}
    )
    graph.add_edge("fix", "execute")  # cycle back
    
    return graph.compile()