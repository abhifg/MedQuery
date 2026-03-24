# MedQuery — Self-Correcting Medical Research Agent

> A Corrective RAG pipeline for medical Q&A built with LangChain, LangGraph, and Groq — retrieves live PubMed literature, self-corrects retrieval quality, and falls back to web search when needed.

---

## What is MedQuery?

MedQuery is an agentic medical Q&A system that goes beyond standard RAG. Instead of blindly injecting retrieved documents into a prompt, it **grades every retrieved document for relevance**, rewrites poor queries, and falls back to web search — all orchestrated as a stateful graph using LangGraph.

---

## Architecture
```
User Query
    ↓
Medical Query Validator  ←── rejects non-medical queries
    ↓
Retrieve from PubMed     ←── live Entrez API + FAISS vector store
    ↓
Grade Documents          ←── LLM-based relevance grading
    ├── relevant   ──────────────────────────► Generate Answer
    └── not relevant → Rewrite Query → Web Search (Tavily) → Generate Answer
                                                  ↓
                                           Final Answer
```

---

## Key Features

- **Corrective RAG** — grades every retrieved PubMed document before generation, not just retrieval
- **Query rewriting** — rewrites poor queries using conversation history before retrying
- **Medical guardrail** — validates and rejects non-medical queries at entry
- **Conversation memory** — maintains context across turns using last exchange
- **LangSmith tracing** — full pipeline observability for every node execution
- **Streamlit UI** — clean chatbot interface with spinner feedback

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (openai/gpt-oss-120b) |
| Orchestration | LangGraph StateGraph |
| RAG Framework | LangChain |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Data Source | PubMed Entrez API |
| Web Search | Tavily |
| UI | Streamlit |
| Observability | LangSmith |

---

## Project Structure
```
ClinIQ/
├── app.py          # Streamlit chatbot UI
├── graph.py        # LangGraph StateGraph definition
├── nodes.py        # All graph node functions + State schema
├── grader.py       # Document grader, hallucination grader, RAG chain
├── retriever.py    # PubMed loader + FAISS vector store builder
├── config.py       # Model names, settings, environment variables
├── .env            # API keys (not committed)
└── requirements.txt
```

---

## How It Works

### 1. Medical Query Validation
Every query is first validated to ensure it's medical or health related. Non-medical queries are rejected immediately with a helpful message.

### 2. Retrieval
The query hits PubMed via Biopython's Entrez API, fetching up to 5 relevant abstracts. These are chunked, embedded using HuggingFace, and stored in a FAISS vector store for similarity search.

### 3. Document Grading (The Corrective Part)
Each retrieved document is graded by an LLM for relevance to the query. If no relevant documents are found, the pipeline triggers query rewriting and falls back to Tavily web search.

### 4. Generation
The answer is generated using only the graded, relevant documents — never from ungrounded knowledge. The RAG chain uses a strict medical prompt that cites sources.

### 5. Conversation Memory
The last user-assistant exchange is passed as context on every turn so follow-up questions like "what are its treatments?" work correctly.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ClinIQ.git
cd MedQuery
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=ls__...
LANGSMITH_PROJECT=ClinIQ
LANGCHAIN_TRACING_V2=true
ENTREZ_EMAIL=your@email.com
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Requirements
```txt
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-tavily
langgraph
streamlit
biopython
faiss-cpu
pydantic
python-dotenv
```

---

## LangSmith Tracing

Every graph run is traced in LangSmith showing:
- Which node executed and in what order
- Latency per node
- Token usage
- Whether web search was triggered
- Document grading decisions

---

## Author

**Abhirup Ghosh**
