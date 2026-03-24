from retriever import load_document,build_retrievers
from typing import List,Annotated
from typing_extensions import TypedDict
from grader import retriever_grader,re_write_chain,tavily_search,rag_chain
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from grader import llm

class State(TypedDict):
    question:str
    generation:str
    web_search:str
    documents:List[str]
    chat_history:Annotated[List,operator.add]

def retrieve(state: State) -> dict:
    print("---retrieving---")

    question = state["question"]
    chat_history = state.get("chat_history", [])
    print("gone to retrieve")
    if chat_history:
        print("got rewritten query")
        rewritten_query = re_write_chain.invoke({
            "question": f"Chat history:\n{chat_history}\n\nUser question:\n{question}"
        })
    else:
        print("not found rewritten query")
        rewritten_query = question
    print(rewritten_query,"first")
    docs = load_document(rewritten_query)
    retriever = build_retrievers(docs)

    if retriever is None:
        print("retriever none")
        return {"question": question, "documents": []}

    documents = retriever.invoke(rewritten_query)
    print(rewritten_query)

    return {
        "question": rewritten_query,
        "documents": documents
    }

def grading_docs(state:State):
    print("---Grading Documents---")
    filtered_docs=[]
    web_search="no"
    question=state["question"]
    docs=state["documents"]
    for d in docs:
        score=retriever_grader.invoke({"document":d.page_content,"question":question})
        if score.binary_score=='yes':
            print("---Document is relevant---")
            filtered_docs.append(d)
        else:
            print("---Document is not relevant---")
            
            continue
    if not filtered_docs:
        web_search="yes"
    return {"question":question,"documents":filtered_docs,"web_search":web_search}

def transform_query(state: State):
    print("---Transforming the query---")

    question = state["question"]

    new_query = re_write_chain.invoke({
        "question": question,
    })

    return {
        "question": new_query,
        "documents": state["documents"]
    }

def web_search(state:State):
    print("---Web Searching initiated---")
    question=state["question"]
    generate=tavily_search.invoke({"query":question})
    documents=generate['results'][0]['content']
    return {"question":question,"documents":documents}

def generate(state: State):
    question = state["question"]
    documents = state["documents"]
    chat_history=state["chat_history"]

    answer = rag_chain.invoke({
        "question": question,
        "context": documents,
    })

    return {
        "question": question,
        "documents": documents,
        "generation": answer,
        "chat_history": [
            ("user", question),
            ("assistant", answer)
        ]
    }

def decide_to_web_search(state:State):
    print("---Decide to Route---")
    question=state["question"]
    documents=state["documents"]
    web_search=state["web_search"]
    if web_search=="yes":
        return "transform_query"
    else:
        return "generate"
   