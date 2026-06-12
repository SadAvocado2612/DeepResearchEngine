import json
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from pipeline.llm_client import call_llm
from pipeline.config import GROQ_MODEL_QUALITY

logger = logging.getLogger("DeepResearchEngine.Verifier")

def evidence_extractor(ranked_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts factual claims and verbatim snippets from the top ranked chunks.
    
    Uses a single batched LLM call for all top-8 chunks to reduce API round-trips.
    
    Inputs:
        ranked_chunks (List[Dict]): Text chunks ranked by VGRH score.
        
    Outputs:
        List[Dict]: Extracted evidence claims with keys: claim, snippet, source, confidence.
    """
    logger.info("Executing evidence_extractor (batched)...")
    if not ranked_chunks:
        return []
        
    # Limit inputs to top 8 chunks to keep context within limits
    input_chunks = []
    for idx, c in enumerate(ranked_chunks[:8]):
        input_chunks.append({
            "chunk_index": idx,
            "text": c["text"][:800],
            "url": c["source_url"]
        })
        
    system_prompt = (
        "You are an expert investigator. Extract the most important concrete claims and evidence items from "
        "ALL of the provided text chunks in a single pass. For each evidence claim, extract:\n"
        "- claim: A clear, summarized factual statement\n"
        "- snippet: The exact text snippet (verbatim) from the source validating the claim\n"
        "- source: The exact URL where the chunk originated\n"
        "- confidence: A confidence float score from 0.0 to 1.0 based on evidence strength\n\n"
        "Return the output as a JSON object with a single key 'evidence' containing the list of all items "
        "extracted from all chunks combined."
    )
    user_prompt = f"Chunks data:\n{json.dumps(input_chunks, indent=2)}"
    
    try:
        response_str = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_QUALITY)
        response_json = json.loads(response_str)
        evidence = response_json.get("evidence", [])
        
        if not isinstance(evidence, list):
            logger.warning("Evidence extractor returned non-list; using empty fallback.")
            evidence = []
        
        # Ensure confidence is normalized between 0.0 and 1.0
        for ev in evidence:
            conf = float(ev.get("confidence", 0.8))
            if conf > 1.0:
                conf = conf / 10.0  # Normalize if LLM outputted on 0-10 scale
            ev["confidence"] = min(max(conf, 0.0), 1.0)
            
        logger.info(f"Evidence extraction: {len(evidence)} claims extracted from 1 LLM call")
        return evidence
    except Exception as e:
        logger.error(f"Evidence extraction failed: {e}")
        return []

def claim_verifier(evidence: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Audits extracted claims against references to flag contradictions or uncertainties.
    
    Uses a single batched LLM call for all claims to reduce API round-trips.
    Boosts confidence by +0.15 if 3+ sources support the claim.
    
    Inputs:
        evidence (List[Dict]): Extracted claims from evidence_extractor.
        chunks (List[Dict]): Full set of retrieved chunks.
        
    Outputs:
        List[Dict]: Verified claims with verification status, confidence, and contradiction notes.
    """
    logger.info("Executing claim_verifier (batched)...")
    if not evidence or not chunks:
        return evidence
        
    # Initialize BM25 search over the full chunk pool for context retrieval
    chunk_texts = [c["text"].lower() for c in chunks]
    tokenized_corpus = [t.split() for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Build a reference pool: for each claim, find top 3 related chunks from OTHER sources
    claims_with_context = []
    for ev in evidence:
        claim = ev["claim"]
        snippet = ev.get("snippet", "")
        source = ev.get("source", "")
        
        tokenized_claim = claim.lower().split()
        scores = bm25.get_scores(tokenized_claim)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        related_chunks = []
        for idx in top_indices:
            candidate = chunks[idx]
            if candidate["source_url"] != source and candidate["text"] not in [rc["text"] for rc in related_chunks]:
                related_chunks.append(candidate)
            if len(related_chunks) >= 3:
                break
                
        # Fill with general top chunks if needed
        if len(related_chunks) < 3:
            for idx in top_indices:
                candidate = chunks[idx]
                if candidate["text"] not in [rc["text"] for rc in related_chunks]:
                    related_chunks.append(candidate)
                if len(related_chunks) >= 3:
                    break
        
        # Keep chunk_context for later
        chunk_context = ""
        for c in chunks:
            if c["source_url"] == source and snippet in c["text"]:
                chunk_context = c["text"]
                break
        if not chunk_context and related_chunks:
            chunk_context = related_chunks[0]["text"]
        
        claims_with_context.append({
            "claim": claim,
            "snippet": snippet,
            "source": source,
            "confidence_orig": ev.get("confidence", 0.8),
            "chunk_context": chunk_context or snippet,
            "references": [{"index": i+1, "url": rc["source_url"], "text": rc["text"][:500]} 
                           for i, rc in enumerate(related_chunks)]
        })
    
    # Single batched LLM call for all claims
    system_prompt = (
        "You are an expert fact-checking auditor. Review ALL provided evidence claims against their supporting snippets "
        "and related reference chunks. For each claim, decide the status:\n"
        "- 'supported': The claim is fully validated by the reference chunks with no conflicts.\n"
        "- 'uncertain': The references do not provide enough clear data to verify the claim.\n"
        "- 'contradicted': The reference chunks contain information that directly contradicts the claim.\n\n"
        "Also assess whether 3 or more independent source URLs empirically support each claim (corroboration).\n\n"
        "Return a JSON object with a single key 'verified_evidence' containing a list of objects, one per claim, "
        "each with: 'claim', 'status' ('supported'|'uncertain'|'contradicted'), 'confidence' (float 0.0-1.0), "
        "'explanation' (reason for status), 'contradiction_note' (string or null), 'corroborated' (true|false)."
    )
    
    claims_input = []
    for i, item in enumerate(claims_with_context):
        claims_input.append({
            "index": i,
            "claim": item["claim"],
            "snippet": item["snippet"],
            "source": item["source"],
            "references": item["references"]
        })
    
    user_prompt = (
        f"Claims to verify:\n{json.dumps(claims_input, indent=2)}\n\n"
        f"Reference Material: See references inside each claim object above."
    )
    
    try:
        response_str = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_QUALITY)
        response_json = json.loads(response_str)
        verified_list = response_json.get("verified_evidence", [])
        
        if not isinstance(verified_list, list):
            logger.warning("Claim verifier returned non-list; falling back to per-claim defaults.")
            raise ValueError("Non-list response")
        
        # Map by index or by claim text
        index_to_result = {}
        for item in verified_list:
            idx = item.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(claims_with_context):
                index_to_result[idx] = item
            else:
                # Try to match by claim text
                for j, ctx in enumerate(claims_with_context):
                    if item.get("claim", "").strip() == ctx["claim"].strip():
                        index_to_result[j] = item
                        break
        
        verified_evidence = []
        supported_count = 0
        uncertain_count = 0
        contradicted_count = 0
        
        for i, ctx in enumerate(claims_with_context):
            rating = index_to_result.get(i, {})
            
            status = rating.get("status", "supported").lower()
            if status not in ["supported", "uncertain", "contradicted"]:
                status = "supported"
            
            conf = float(rating.get("confidence", ctx["confidence_orig"]))
            if conf > 1.0:
                conf = conf / 10.0
            conf = min(max(conf, 0.0), 1.0)
            
            corroborated = rating.get("corroborated", False)
            if corroborated and status == "supported":
                conf = min(conf + 0.15, 1.0)
            
            explanation = rating.get("explanation", "Matches references.")
            contradiction_note = rating.get("contradiction_note", None)
            if status != "contradicted":
                contradiction_note = None
            
            if status == "supported":
                supported_count += 1
            elif status == "uncertain":
                uncertain_count += 1
            elif status == "contradicted":
                contradicted_count += 1
            
            verified_evidence.append({
                "claim": ctx["claim"],
                "snippet": ctx["snippet"],
                "source": ctx["source"],
                "status": status,
                "confidence": round(conf, 2),
                "explanation": explanation,
                "contradiction_note": contradiction_note,
                "corroborated": corroborated,
                "chunk_context": ctx["chunk_context"]
            })
        
        logger.info(f"Claim verification: {supported_count} supported, {uncertain_count} uncertain, {contradicted_count} contradicted")
        return verified_evidence
        
    except Exception as e:
        logger.error(f"Batched claim verification failed: {e}. Falling back to defaults.")
        # Safe fallback — return all claims as supported with original confidence
        verified_evidence = []
        for ctx in claims_with_context:
            verified_evidence.append({
                "claim": ctx["claim"],
                "snippet": ctx["snippet"],
                "source": ctx["source"],
                "status": "supported",
                "confidence": round(ctx["confidence_orig"], 2),
                "explanation": "Default supported due to verification error.",
                "contradiction_note": None,
                "corroborated": False,
                "chunk_context": ctx["chunk_context"]
            })
        return verified_evidence
