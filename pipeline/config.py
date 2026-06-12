import os

# Central Configuration for Deep Research Engine

# Models
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST    = "llama-3.1-8b-instant"
GROQ_MODEL_QUALITY = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking & Overlap
CHUNK_SIZE = 512       # Approximate word count per chunk
CHUNK_OVERLAP = 50     # Overlap word count between consecutive chunks

# Retrieval settings
BM25_TOP_K = 20        # Top candidates from BM25
VECTOR_TOP_K = 20      # Top candidates from Vector similarity
HYBRID_TOP_K = 20      # Top candidates to pass to VGRH ranker after RRF (raised from 15)
RRF_K = 60             # Constant parameter for Reciprocal Rank Fusion

# VGRH Weights
WEIGHT_V = 0.3
WEIGHT_G = 0.25
WEIGHT_R = 0.3
WEIGHT_H = 0.15

# Gap detection and iterations
MAX_ITERATIONS = 2      # Maximum deep research gap loop runs
MAX_SOURCES = 8         # Maximum source URLs fetched per search round
MAX_SOURCES_FETCH = 12  # Hard cap on sources to fetch after authority-based pre-ranking
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Domain Authority Lists
AUTHORITY_BOOST_DOMAINS = [
    ".edu", ".gov", ".org", "arxiv.org", "pubmed", 
    "nature.com", "scholar.google", "reuters.com", 
    "bbc.com", "who.int"
]

AUTHORITY_PENALTY_DOMAINS = [
    "reddit.com", "quora.com", "yahoo answers", "answers.yahoo.com"
]
