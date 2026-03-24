import os
from dotenv import load_dotenv
load_dotenv()
from Bio import Entrez
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import ENTREZ_EMAIL,EMBEDDING_MODEL

Entrez.email=ENTREZ_EMAIL

def load_document(query:str,max_docs:int=5)->list[Document]:
    handle=Entrez.esearch(db="pubmed",term=query,retmax=max_docs)
    ids=Entrez.read(handle)["IdList"]
    
    if not ids:
        return []
    
    fetch_handle=Entrez.efetch(db="pubmed",id=",".join(ids),retmode="xml",rettype="xml")

    records=Entrez.read(fetch_handle)

    docs=[]

    for record in records["PubmedArticle"]:
        article=record["MedlineCitation"]["Article"]
        title=str(article.get("ArticleTitle",""))
        abstract=" ".join(article.get("Abstract",{}).get("AbstractText",[""]))
        pmid=str(record["MedlineCitation"]["PMID"])

        docs.append(Document(
            page_content=f"{title}\n\n{abstract}",
            metadata={
                "title":title,
                "pmid":pmid,
                "source":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            }
        ))
    
    return docs

def build_retrievers(docs:list[Document]):
    if not docs:
        return None
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    splitted_docs=splitter.split_documents(docs)
    embedding=HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    vectorstore=FAISS.from_documents(documents=splitted_docs,embedding=embedding)

    retriever=vectorstore.as_retriever()
    return retriever

