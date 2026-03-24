from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
from config import LLM_MODEL
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

llm=ChatGroq(model=LLM_MODEL)

class Document_Grader(BaseModel):
    binary_score:Literal["yes","no"]=Field(description="Grade the documents with yes or no")

doc_grader=llm.with_structured_output(Document_Grader)

sys_prompt="""You are a Document Grader Expert. If the retrieved document contains relevance or semantic meaning with the\n
user's question then please grade the retrieved document with 'yes' and if the retrieved documents doesn't contain any relevance or semantic meaning\n
then please grade the document with 'no'."""

grader_prompt=ChatPromptTemplate.from_messages([
    ("system",sys_prompt),
    ("human","Here is the retrieved documents : {document}\n and User's question : {question}")
])
retriever_grader=grader_prompt|doc_grader

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{question}
If the question is not related to medical/health/treatment/wellness related then simply say 'Please ask me medical related questions'\n
Don't answer those questions which are not medical related even if you are asked so.
Answer clearly and accurately within 2 to 3 lines not more.
""")
rag_chain=rag_prompt|llm|StrOutputParser()

rewrite_prompt=""" You are a question rewriter that converts an user's question to a better version based on chat history that will be used for web search.\n
Just look at the input and try to convert it into a better versioning with the current context.
"""
re_write_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",rewrite_prompt),
        ("human","Here is the question: \n {question} \n formulate a improved version .")
    ]
)
re_write_chain=re_write_prompt|llm|StrOutputParser()

tavily_search=TavilySearch(max_results=1)


