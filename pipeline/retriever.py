import os
import io
import asyncio
import logging
import httpx
import numpy as np
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from pipeline.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BM25_TOP_K,
    VECTOR_TOP_K,
    HYBRID_TOP_K,
    RRF_K,
    MAX_SOURCES,
    MAX_SOURCES_FETCH,
    EMBEDDING_MODEL_NAME,
    AUTHORITY_BOOST_DOMAINS,
    AUTHORITY_PENALTY_DOMAINS
)

logger = logging.getLogger("DeepResearchEngine.Retriever")

# Global lazy loading of sentence-transformers embedding model
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Lazily loads and returns the SentenceTransformer model on CPU."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on CPU...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _embedding_model

# -----------------------------------------------------------------------------
# Source Authority Pre-Ranker (for pre-fetch filtering)
# -----------------------------------------------------------------------------
def quick_authority_score(url: str) -> float:
    """Returns a simple numeric authority score for pre-fetch source filtering."""
    url_lower = url.lower()
    academic_indicators = [".edu", "arxiv.org", "pubmed", "nature.com", "scholar.google"]
    gov_indicators = [".gov", "who.int"]
    news_indicators = ["reuters.com", "bbc.com"]
    penalty_domains = ["reddit.com", "quora.com", "yahoo answers", "answers.yahoo.com"]
    
    if any(ind in url_lower for ind in academic_indicators):
        return 2.0
    if any(ind in url_lower for ind in gov_indicators):
        return 2.0
    if any(ind in url_lower for ind in news_indicators):
        return 1.5
    if any(pen in url_lower for pen in penalty_domains):
        return 0.0
    return 1.0

# -----------------------------------------------------------------------------
# Source Discoverer (Tavily or DuckDuckGo)
# -----------------------------------------------------------------------------
async def search_tavily(query: str, api_key: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Runs a query on Tavily Search API.
    
    Inputs:
        query (str): The search string.
        api_key (str): Tavily API credential.
        max_results (int): Max links to return.
        
    Outputs:
        List of dicts: {'url', 'title', 'snippet'}.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=8.0)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "url": item.get("url"),
                        "title": item.get("title", "Untitled Webpage"),
                        "snippet": item.get("content", "")
                    })
                return results
    except Exception as e:
        logger.error(f"Tavily search error for '{query}': {e}")
    return []

def search_ddg(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Runs a query on DuckDuckGo Search.
    
    Inputs:
        query (str): The search string.
        max_results (int): Max links to return.
        
    Outputs:
        List of dicts: {'url', 'title', 'snippet'}.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "url": r.get("href"),
                    "title": r.get("title", "Untitled Webpage"),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        logger.error(f"DuckDuckGo search error for '{query}': {e}")
    return results

async def source_discoverer(
    search_queries: List[str],
    tavily_key: str = None,
    iteration: int = 1,
    sub_question_indices: List[int] = None
) -> List[Dict[str, Any]]:
    """Queries search indexes to retrieve unique sources.
    
    Inputs:
        search_queries (List[str]): Optimized keywords to run.
        tavily_key (str): API key for Tavily (optional fallback to DDG).
        iteration (int): The iteration index (1 or 2).
        sub_question_indices (List[int]): Corresponding sub-question index per query.
        
    Outputs:
        List of sources with URL, Title, type (pdf/web), iteration, and sub_question_index.
    """
    logger.info(f"Executing source_discoverer for iteration {iteration}...")
    all_results = []
    
    # Map each query to its sub_question_index if provided
    query_sq_map = {}
    if sub_question_indices and len(sub_question_indices) == len(search_queries):
        for i, q in enumerate(search_queries):
            query_sq_map[q] = sub_question_indices[i]
    
    if tavily_key:
        tasks = [search_tavily(q, tavily_key) for q in search_queries[:8]]
        searches = await asyncio.gather(*tasks)
        for q_idx, r_list in enumerate(searches):
            sq_idx = query_sq_map.get(search_queries[q_idx] if q_idx < len(search_queries) else "", -1)
            for res in r_list:
                res["sq_idx"] = sq_idx
            all_results.extend(r_list)
    else:
        for q_idx, q in enumerate(search_queries[:4]):  # Limit queries to prevent DDG block
            sq_idx = query_sq_map.get(q, -1)
            results = search_ddg(q)
            for res in results:
                res["sq_idx"] = sq_idx
            all_results.extend(results)
            await asyncio.sleep(1.0)
            
    seen_urls = set()
    unique_sources = []
    for res in all_results:
        url = res["url"]
        if url and url not in seen_urls:
            seen_urls.add(url)
            url_lower = url.lower()
            source_type = "pdf" if url_lower.endswith(".pdf") else "web"
            unique_sources.append({
                "url": url,
                "title": res["title"],
                "type": source_type,
                "snippet": res["snippet"],
                "iteration": iteration,
                "sq_idx": res.get("sq_idx", -1),
                "_authority_score": quick_authority_score(url)
            })
    
    # If more than MAX_SOURCES_FETCH discovered, sort by authority and keep top MAX_SOURCES_FETCH
    if len(unique_sources) > MAX_SOURCES_FETCH:
        unique_sources.sort(key=lambda x: x["_authority_score"], reverse=True)
        unique_sources = unique_sources[:MAX_SOURCES_FETCH]
        logger.info(f"Capped to {MAX_SOURCES_FETCH} sources after authority-based pre-ranking.")
        
    # Remove internal score from output
    for src in unique_sources:
        src.pop("_authority_score", None)
            
    return unique_sources[:MAX_SOURCES]

# -----------------------------------------------------------------------------
# Web & PDF Fetcher
# -----------------------------------------------------------------------------
async def fetch_single_source(client: httpx.AsyncClient, source: Dict[str, Any]) -> Dict[str, Any]:
    """Downloads content from a webpage URL or PDF endpoint.
    
    Inputs:
        client: Async HTTP Client.
        source: Dictionary containing source details.
        
    Outputs:
        Dictionary with content text/bytes, type, and success status.
    """
    url = source["url"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = await client.get(url, headers=headers, follow_redirects=True, timeout=8.0)
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "").lower()
            is_pdf = "application/pdf" in content_type or url.lower().endswith(".pdf")
            return {
                **source,
                "content": response.content if is_pdf else response.text,
                "type": "pdf" if is_pdf else "web",
                "success": True
            }
        return {**source, "content": f"Failed with HTTP {response.status_code}", "success": False}
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return {**source, "content": f"Error during fetch: {str(e)}", "success": False}

async def web_pdf_fetcher(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concurrently fetches documents from discovered sources.
    
    Inputs:
        sources: Discovered URLs.
        
    Outputs:
        Raw fetched document strings/bytes.
    """
    logger.info(f"Executing web_pdf_fetcher — {len(sources)} concurrent requests, timeout 8s...")
    async with httpx.AsyncClient() as client:
        tasks = [fetch_single_source(client, src) for src in sources]
        results = await asyncio.gather(*tasks)
    return results

# -----------------------------------------------------------------------------
# Parser & Chunker
# -----------------------------------------------------------------------------
def clean_html(html_str: str) -> str:
    """Extracts raw text content from HTML via BeautifulSoup4."""
    try:
        soup = BeautifulSoup(html_str, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.error(f"HTML cleanup error: {e}")
        return html_str

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Parses text pages out of PDF stream using PyPDF."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"PDF extract error: {e}")
        return ""

def parser_chunker(raw_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts fetched documents into tokenized text chunks.
    
    Inputs:
        raw_documents: Fetched document strings/bytes.
        
    Outputs:
        List of chunks with word index segments and metadata.
        Each chunk carries sq_idx from its source document.
    """
    logger.info("Executing parser_chunker...")
    chunks = []
    
    for doc in raw_documents:
        if not doc.get("success", False):
            continue
            
        if doc["type"] == "pdf":
            parsed_text = extract_pdf_text(doc["content"])
        else:
            parsed_text = clean_html(doc["content"])
            
        if not parsed_text or len(parsed_text.strip()) < 50:
            continue
            
        words = parsed_text.split()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        sq_idx = doc.get("sq_idx", -1)
        
        chunk_index = 0
        if len(words) <= CHUNK_SIZE:
            chunks.append({
                "text": " ".join(words),
                "metadata": {"title": doc["title"], "iteration": doc.get("iteration", 1)},
                "source_url": doc["url"],
                "chunk_id": f"{doc['url']}#chunk_{chunk_index}",
                "sq_idx": sq_idx
            })
            continue
            
        for i in range(0, len(words), step):
            window = words[i:i + CHUNK_SIZE]
            if not window:
                break
            chunks.append({
                "text": " ".join(window),
                "metadata": {"title": doc["title"], "iteration": doc.get("iteration", 1)},
                "source_url": doc["url"],
                "chunk_id": f"{doc['url']}#chunk_{chunk_index}",
                "sq_idx": sq_idx
            })
            chunk_index += 1
            if i + CHUNK_SIZE >= len(words):
                break
                
    logger.info(f"Generated {len(chunks)} text chunks from sources.")
    return chunks

# -----------------------------------------------------------------------------
# Hybrid Retrieval (BM25 + Vector Search + RRF)
# -----------------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between a vector and a matrix of vectors."""
    dot_product = np.dot(v2, v1)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    return dot_product / (norm_v1 * norm_v2 + 1e-9)

def hybrid_retriever(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Retrieves top chunks using reciprocal rank fusion of BM25 and Vector Search.
    
    Deduplicates by source URL (keeps best-scoring chunk per URL) before returning.
    
    Inputs:
        chunks: All sliding-window text chunks.
        query: Research topic query string.
        
    Outputs:
        Top hybrid retrieved chunks with BM25 score, similarity, RRF score, and retrieval method.
    """
    logger.info("Executing hybrid_retriever (BM25 + Vector + RRF)...")
    if not chunks:
        return []
        
    # --- 1. BM25 Search ---
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top BM25 candidate indices
    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    bm25_top = [chunks[idx] for idx in bm25_indices]
    logger.info(f"BM25 retrieval: top {len(bm25_top)} chunks selected")
    
    # --- 2. Vector Search (batched) ---
    model = get_embedding_model()
    chunk_texts = [c["text"] for c in chunks]
    # Single batched encode call — much faster than one-at-a-time
    chunk_embeddings = model.encode(chunk_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    vector_similarities = cosine_similarity(query_embedding, chunk_embeddings)
    
    # Get top Vector candidate indices
    vector_indices = np.argsort(vector_similarities)[::-1][:VECTOR_TOP_K]
    vector_top = [chunks[idx] for idx in vector_indices]
    logger.info(f"Vector retrieval: top {len(vector_top)} chunks selected")
    
    # --- 3. Reciprocal Rank Fusion (RRF) ---
    # Index lookups
    bm25_ranks = {item["chunk_id"]: rank for rank, item in enumerate(bm25_top, 1)}
    vector_ranks = {item["chunk_id"]: rank for rank, item in enumerate(vector_top, 1)}
    
    # Calculate RRF scores for any chunk retrieved in either top list
    all_retrieved_ids = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
    rrf_scored_chunks = []
    
    # Map chunk_id to actual chunk dict
    id_to_chunk = {c["chunk_id"]: c for c in chunks}
    # Build index map for bm25/vector scores
    chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    
    for cid in all_retrieved_ids:
        chunk = id_to_chunk[cid]
        bm_rank = bm25_ranks.get(cid)
        vec_rank = vector_ranks.get(cid)
        
        rrf_score = 0.0
        method = "None"
        
        if bm_rank is not None and vec_rank is not None:
            rrf_score = 1.0 / (RRF_K + bm_rank) + 1.0 / (RRF_K + vec_rank)
            method = "Both"
        elif bm_rank is not None:
            rrf_score = 1.0 / (RRF_K + bm_rank)
            method = "BM25"
        elif vec_rank is not None:
            rrf_score = 1.0 / (RRF_K + vec_rank)
            method = "Vector"
            
        chunk_idx = chunk_id_to_idx.get(cid, 0)
        chunk_copy = dict(chunk)
        chunk_copy["rrf_score"] = float(rrf_score)
        chunk_copy["retrieval_method"] = method
        chunk_copy["bm25_score"] = float(bm25_scores[chunk_idx])
        chunk_copy["vector_score"] = float(vector_similarities[chunk_idx])
        
        rrf_scored_chunks.append(chunk_copy)
    
    logger.info(f"RRF merge: {len(rrf_scored_chunks)} unique chunks after deduplication")
        
    # Sort descending by RRF score
    rrf_scored_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # Deduplicate by source_url — keep only the best-scoring chunk per URL
    seen_urls = set()
    deduped_chunks = []
    for chunk in rrf_scored_chunks:
        url = chunk["source_url"]
        if url not in seen_urls:
            seen_urls.add(url)
            deduped_chunks.append(chunk)
    
    top_hybrid_chunks = deduped_chunks[:HYBRID_TOP_K]
    
    logger.info(f"Hybrid retrieval finished. Kept top {len(top_hybrid_chunks)} chunks after URL deduplication.")
    return top_hybrid_chunks
