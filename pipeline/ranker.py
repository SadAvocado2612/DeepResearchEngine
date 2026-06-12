import json
import logging
from typing import List, Dict, Any
from pipeline.config import (
    WEIGHT_V,
    WEIGHT_G,
    WEIGHT_R,
    WEIGHT_H,
    AUTHORITY_BOOST_DOMAINS,
    AUTHORITY_PENALTY_DOMAINS,
    GROQ_MODEL_FAST
)
from pipeline.llm_client import call_llm

logger = logging.getLogger("DeepResearchEngine.Ranker")

def source_authority_scorer(url: str) -> tuple[float, str]:
    """Calculates Veracity boost/penalty and returns the source authority tier.
    
    Inputs:
        url (str): The URL of the source.
        
    Outputs:
        tuple[float, str]: (veracity_modifier, authority_tier) — tier uses plain text labels.
    """
    url_lower = url.lower()
    
    # 1. Academic Tier
    academic_indicators = [".edu", "arxiv.org", "pubmed", "nature.com", "scholar.google"]
    if any(ind in url_lower for ind in academic_indicators):
        return 0.10, "Academic"
        
    # 2. Government Tier
    gov_indicators = [".gov", "who.int"]
    if any(ind in url_lower for ind in gov_indicators):
        return 0.10, "Government"
        
    # 3. News Tier
    news_indicators = ["reuters.com", "bbc.com"]
    if any(ind in url_lower for ind in news_indicators):
        return 0.10, "News"
        
    # 4. Penalty domains
    if any(pen in url_lower for pen in AUTHORITY_PENALTY_DOMAINS):
        return -0.05, "Web"
        
    return 0.0, "Web"

def vgrh_ranker(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Evaluates candidates on Veracity, Grounding, Relevance, and Helpfulness.
    
    This function processes chunks in batches of 5, calls the LLM for multi-criteria scores,
    applies the source authority modifications, and calculates the final weighted VGRH score.
    
    Inputs:
        chunks (List[Dict]): Chunks retrieved by hybrid retriever.
        query (str): Primary user query.
        
    Outputs:
        List[Dict]: Chunks sorted by final weighted VGRH score descending.
    """
    logger.info("Executing vgrh_ranker with authority scoring...")
    if not chunks:
        return []
        
    ranked_chunks = []
    batch_size = 5
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_input = []
        for idx, item in enumerate(batch):
            batch_input.append({
                "batch_index": idx,
                "title": item["metadata"].get("title", "Untitled Source"),
                "url": item["source_url"],
                "text": item["text"][:1000]
            })
            
        system_prompt = (
            "You are an expert factual data evaluator rating text blocks relative to a research topic. "
            "Evaluate each block of text on these four parameters (ratings must be floats between 0.0 and 10.0):\n"
            "1. Veracity (V): The factual correctness, authoritativeness, and consensus validity.\n"
            "2. Grounding (G): The level of grounding in empirical data, citations, or logical proofs.\n"
            "3. Relevance (R): How directly relevant the text is to solving the query.\n"
            "4. Helpfulness (H): How rich, readable, and structured the insights are.\n\n"
            "You must return the scores as a JSON object containing a 'ratings' list. "
            "Each item in the list must specify 'batch_index', 'v', 'g', 'r', 'h', and 'reason' (explanation)."
        )
        
        user_prompt = f"Research Query: {query}\n\nChunks:\n{json.dumps(batch_input, indent=2)}"
        
        try:
            response_str = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_FAST)
            response_json = json.loads(response_str)
            ratings = {item["batch_index"]: item for item in response_json.get("ratings", [])}
        except Exception as e:
            logger.error(f"Error rating batch: {e}")
            ratings = {}
            
        for idx, item in enumerate(batch):
            item_copy = dict(item)
            rating = ratings.get(idx, {"v": 5.0, "g": 5.0, "r": 5.0, "h": 5.0, "reason": "Evaluator defaulted score."})
            
            v = float(rating.get("v", 5.0))
            g = float(rating.get("g", 5.0))
            r = float(rating.get("r", 5.0))
            h = float(rating.get("h", 5.0))
            reason = rating.get("reason", "Relevant source chunk.")
            
            # --- Source Authority Scorer ---
            v_mod, auth_tier = source_authority_scorer(item["source_url"])
            # Apply veracity modification: V is 0-10, so mod is multiplied by 10 (i.e. +1.0 or -0.5)
            v_adjusted = min(max(v + (v_mod * 10.0), 0.0), 10.0)
            
            # Apply VGRH weighted score: V=0.3, G=0.25, R=0.3, H=0.15
            weighted_score = round(v_adjusted * WEIGHT_V + g * WEIGHT_G + r * WEIGHT_R + h * WEIGHT_H, 2)
            
            item_copy["v"] = v_adjusted
            item_copy["g"] = g
            item_copy["r"] = r
            item_copy["h"] = h
            item_copy["v_original"] = v
            item_copy["authority_tier"] = auth_tier
            item_copy["reason"] = reason
            item_copy["vgrh_score"] = weighted_score
            ranked_chunks.append(item_copy)
            
    # Sort descending by VGRH score
    ranked_chunks.sort(key=lambda x: x["vgrh_score"], reverse=True)
    return ranked_chunks
