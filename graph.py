from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from nodes import (
    State,
    retrieve,
    grading_docs,
    transform_query,
    web_search,
    generate,
    decide_to_web_search
)

workflow=StateGraph(State)

workflow.add_node("Retrieve",retrieve)
workflow.add_node("Grade",grading_docs)
workflow.add_node("Rewrite",transform_query)
workflow.add_node("Web_Search",web_search)
workflow.add_node("Generate",generate)

workflow.add_edge(START,"Retrieve")
workflow.add_edge("Retrieve","Grade")
workflow.add_conditional_edges(
    "Grade",
    decide_to_web_search,
    {
        "transform_query":"Rewrite",
        "generate":"Generate"
    }
)
workflow.add_edge("Rewrite","Web_Search")
workflow.add_edge("Web_Search","Generate")
workflow.add_edge("Generate",END)

memory=MemorySaver()
graph=workflow.compile(checkpointer=memory)