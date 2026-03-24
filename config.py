import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "MedQuery"

# Model settings
LLM_MODEL        = "openai/gpt-oss-safeguard-20b"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K            = 4
MAX_PUBMED_DOCS  = 5
MAX_RETRIES      = 2

# Chunking settings
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

# Entrez email (not secret, fine to read here)
ENTREZ_EMAIL     = os.getenv("ENTREZ_EMAIL")