import os
import json
import logging
import asyncio
import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import pipeline sub-modules
from pipeline.config import MAX_ITERATIONS, TAVILY_KEY
from pipeline.planner import research_planner, query_generator, gap_detector
from pipeline.retriever import source_discoverer, web_pdf_fetcher, parser_chunker, hybrid_retriever
from pipeline.ranker import vgrh_ranker
from pipeline.verifier import evidence_extractor, claim_verifier
from pipeline.reporter import report_generator, compile_self_contained_html

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DeepResearchEngine.Main")

app = FastAPI(title="Agentic Deep Research Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main single page dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/research")
async def run_research_sse(query: str = Query(..., description="The research topic query")):
    """Streams research steps and final results using Server-Sent Events (SSE)."""
    
    async def sse_generator():
        # Timers tracking dict for the stepper UI: {step_id: start_time}
        timers = {}
        pipeline_start = asyncio.get_event_loop().time()
        
        def now_ts() -> str:
            return datetime.datetime.now().strftime("%H:%M:%S")
        
        def log_event(message: str) -> str:
            """Emit a log event with timestamp."""
            return f"data: {json.dumps({'event': 'log', 'time': now_ts(), 'message': message})}\n\n"
        
        def start_step(step_id: str, message: str):
            timers[step_id] = asyncio.get_event_loop().time()
            return f"data: {json.dumps({'type': 'step_start', 'step': step_id, 'message': message})}\n\n"
            
        def complete_step(step_id: str, message: str):
            elapsed = 0.0
            if step_id in timers:
                elapsed = round(asyncio.get_event_loop().time() - timers[step_id], 1)
            return f"data: {json.dumps({'type': 'step_complete', 'step': step_id, 'message': message, 'elapsed': elapsed})}\n\n"
            
        def fail_step(step_id: str, message: str):
            elapsed = 0.0
            if step_id in timers:
                elapsed = round(asyncio.get_event_loop().time() - timers[step_id], 1)
            return f"data: {json.dumps({'type': 'step_failed', 'step': step_id, 'message': message, 'elapsed': elapsed})}\n\n"
        
        def subquestion_update(sq_index: int, sources_found: int, status: str, was_gap: bool = False) -> str:
            return f"data: {json.dumps({'type': 'subquestion_update', 'index': sq_index, 'sources_found': sources_found, 'status': status, 'was_gap': was_gap})}\n\n"

        yield log_event("Pipeline started")
        await asyncio.sleep(0.1)

        try:
            # --- 1. Research Planning ---
            yield start_step("step-planning", "Decomposing query to sub-questions")
            yield log_event("Running research planner...")
            
            sub_questions, plan_md = research_planner(query)
            
            yield f"data: {json.dumps({'type': 'plan_generated', 'sub_questions': sub_questions, 'plan_md': plan_md})}\n\n"
            yield log_event(f"Research planner decomposed query into {len(sub_questions)} sub-questions")
            yield complete_step("step-planning", "Research plan generated")
            await asyncio.sleep(0.1)

            # Define main registers for loop aggregation
            all_sources = []
            all_chunks = []
            iteration = 1
            current_sub_questions = list(sub_questions)
            
            # Build mapping from sub-question text to its index in the original list
            sq_text_to_idx = {q: i for i, q in enumerate(sub_questions)}
            
            # Track sources per sub-question: {sq_idx: set of URLs}
            sq_source_map: Dict[int, set] = {i: set() for i in range(len(sub_questions))}
            
            verified_evidence = []

            # --- Iterative Research Loop ---
            while iteration <= MAX_ITERATIONS and len(current_sub_questions) > 0:
                iter_prefix = f"Iteration {iteration}/{MAX_ITERATIONS}"
                yield log_event(f"{iter_prefix} — searching {len(current_sub_questions)} sub-questions")
                
                # Update sub-question card status in UI (searching)
                yield f"data: {json.dumps({'type': 'plan_status_update', 'questions': current_sub_questions, 'status': 'searching'})}\n\n"

                # --- 2. Query Generation ---
                yield start_step("step-queries", f"Generating search terms for iteration {iteration}")
                queries = query_generator(current_sub_questions)
                yield log_event(f"Query generator produced {len(queries)} search queries")
                yield complete_step("step-queries", "Query generation complete")
                await asyncio.sleep(0.1)

                # Build sub-question index per query — distribute queries evenly across active sub-questions
                sq_indices_for_queries = []
                for q_idx, _ in enumerate(queries):
                    sq_text = current_sub_questions[q_idx % len(current_sub_questions)]
                    sq_idx = sq_text_to_idx.get(sq_text, -1)
                    sq_indices_for_queries.append(sq_idx)

                # --- 3. Source Discovery ---
                yield start_step("step-discover", f"Crawling search engines (Iteration {iteration})")
                discovered_sources = await source_discoverer(
                    queries, TAVILY_KEY, iteration, sq_indices_for_queries
                )
                yield log_event(f"Discovered {len(discovered_sources)} candidate sources")
                
                # Append unique sources to list and update sq_source_map
                new_sources_count = 0
                for src in discovered_sources:
                    if src["url"] not in [s["url"] for s in all_sources]:
                        all_sources.append(src)
                        new_sources_count += 1
                    sq_i = src.get("sq_idx", -1)
                    if sq_i >= 0:
                        sq_source_map.setdefault(sq_i, set()).add(src["url"])
                        
                yield complete_step("step-discover", "Source discovery complete")
                await asyncio.sleep(0.1)

                # --- 4. Web & PDF Fetching ---
                yield start_step("step-fetching", f"Fetching document contents (Iteration {iteration})")
                yield log_event(f"Fetching sources — {len(discovered_sources)} concurrent requests, timeout 8s")
                
                fetched_docs = await web_pdf_fetcher(discovered_sources)
                success_count = sum(1 for d in fetched_docs if d.get("success", False))
                failed_count = len(discovered_sources) - success_count
                yield log_event(f"Fetched {success_count} of {len(discovered_sources)} sources successfully ({failed_count} timed out or failed)")
                
                # --- 5. Parsing & Chunking ---
                chunks = parser_chunker(fetched_docs)
                all_chunks.extend(chunks)
                yield log_event(f"Chunked {success_count} documents into {len(chunks)} chunks")
                yield complete_step("step-fetching", "Content fetching & chunking complete")
                await asyncio.sleep(0.1)

                # --- 6. Hybrid Retrieval ---
                yield start_step("step-retrieval", "Running hybrid keyword + vector retriever")
                yield log_event(f"Running BM25 + vector (all-MiniLM-L6-v2) hybrid retrieval on {len(all_chunks)} chunks...")
                
                top_chunks = hybrid_retriever(all_chunks, query)
                
                # Log retrieval stats from chunk data
                bm25_only = sum(1 for c in top_chunks if c.get("retrieval_method") == "BM25")
                vec_only = sum(1 for c in top_chunks if c.get("retrieval_method") == "Vector")
                both = sum(1 for c in top_chunks if c.get("retrieval_method") == "Both")
                yield log_event(f"BM25 retrieval: top {bm25_only + both} chunks | Vector retrieval: top {vec_only + both} chunks | RRF merge: {len(top_chunks)} unique chunks after deduplication")
                yield complete_step("step-retrieval", "Hybrid retrieval complete")
                await asyncio.sleep(0.1)

                # --- 7. VGRH Ranking ---
                yield start_step("step-ranking", "Evaluating Veracity, Grounding, Relevance & Helpfulness")
                yield log_event(f"VGRH ranking {len(top_chunks)} chunks with authority scoring...")
                
                ranked_chunks = vgrh_ranker(top_chunks, query)
                yield log_event(f"VGRH ranking complete — top {len(ranked_chunks)} chunks selected")
                yield complete_step("step-ranking", "VGRH ranking complete")
                await asyncio.sleep(0.1)

                # --- 8. Evidence Extraction ---
                yield start_step("step-evidence", "Extracting concrete facts & snippets")
                yield log_event("Mining top-8 chunks for claims and source quotes (1 batched LLM call)...")
                
                evidence = evidence_extractor(ranked_chunks)
                yield log_event(f"Evidence extraction: {len(evidence)} claims extracted from 1 LLM call")
                yield complete_step("step-evidence", "Evidence extraction complete")
                await asyncio.sleep(0.1)

                # --- 9. Claim Verification ---
                yield start_step("step-verification", "Auditing claims against reference materials")
                yield log_event(f"Verifying {len(evidence)} claims for support, uncertainty, contradictions (1 batched LLM call)...")
                
                verified_evidence = claim_verifier(evidence, ranked_chunks)
                
                supported_n = sum(1 for e in verified_evidence if e.get("status") == "supported")
                uncertain_n = sum(1 for e in verified_evidence if e.get("status") == "uncertain")
                contradicted_n = sum(1 for e in verified_evidence if e.get("status") == "contradicted")
                yield log_event(f"Claim verification: {supported_n} supported, {uncertain_n} uncertain, {contradicted_n} contradicted")
                yield complete_step("step-verification", "Claim verification complete")
                await asyncio.sleep(0.1)

                # --- Send per-sub-question source count updates ---
                # Map sources from ranked_chunks back to sub-questions
                for chunk in ranked_chunks:
                    url = chunk["source_url"]
                    sq_i = chunk.get("sq_idx", -1)
                    if sq_i >= 0:
                        sq_source_map.setdefault(sq_i, set()).add(url)
                
                # Emit subquestion_update for each active sub-question
                for sq_text in current_sub_questions:
                    sq_i = sq_text_to_idx.get(sq_text, -1)
                    if sq_i >= 0:
                        src_count = len(sq_source_map.get(sq_i, set()))
                        yield subquestion_update(sq_i, src_count, "answered")

                # Also keep plan_status_update for backward compat
                yield f"data: {json.dumps({'type': 'plan_status_update', 'questions': current_sub_questions, 'status': 'answered'})}\n\n"

                # --- Gap Detection Step ---
                if iteration < MAX_ITERATIONS:
                    yield log_event("Running gap detection...")
                    gaps = gap_detector(query, sub_questions, verified_evidence)
                    if gaps:
                        yield log_event(f"Gap detection: {len(gaps)} sub-questions have weak evidence — running follow-up search")
                        yield f"data: {json.dumps({'type': 'plan_status_update', 'questions': gaps, 'status': 'gap'})}\n\n"
                        # Mark gap sub-questions in subquestion_update
                        for sq_text in gaps:
                            sq_i = sq_text_to_idx.get(sq_text, -1)
                            if sq_i >= 0:
                                src_count = len(sq_source_map.get(sq_i, set()))
                                yield subquestion_update(sq_i, src_count, "gap")
                        current_sub_questions = gaps
                        iteration += 1
                    else:
                        yield log_event("Gap detection: no research gaps found — proceeding to report")
                        current_sub_questions = []
                else:
                    break
            
            # Mark any sub-questions that ended up with 0 sources as unanswered
            for sq_text in sub_questions:
                sq_i = sq_text_to_idx.get(sq_text, -1)
                if sq_i >= 0:
                    src_count = len(sq_source_map.get(sq_i, set()))
                    if src_count == 0:
                        yield subquestion_update(sq_i, 0, "unanswered")

            yield log_event("All gaps resolved — generating final report")

            # --- 10. Report Generation ---
            yield start_step("step-report", "Synthesizing final cited document")
            yield log_event("Compiling Markdown report body and citation tables...")
            
            # Map RRF retrieval scores and authority values back to sources table
            url_scores = {}
            for chunk in ranked_chunks:
                url = chunk["source_url"]
                if url not in url_scores:
                    url_scores[url] = {
                        "vgrh_score": chunk.get("vgrh_score", 5.0),
                        "v": chunk.get("v", 5.0),
                        "g": chunk.get("g", 5.0),
                        "r": chunk.get("r", 5.0),
                        "h": chunk.get("h", 5.0),
                        "authority_tier": chunk.get("authority_tier", "Web"),
                        "retrieval_method": chunk.get("retrieval_method", "BM25"),
                        "reason": chunk.get("reason", "Highly ranked reference block.")
                    }
            
            updated_sources = []
            for src in all_sources:
                metrics = url_scores.get(src["url"], {
                    "vgrh_score": 0.0, "v": 0.0, "g": 0.0, "r": 0.0, "h": 0.0, 
                    "authority_tier": "Web", "retrieval_method": "BM25", 
                    "reason": "Source not retrieved in final top chunks."
                })
                # Retain original iteration
                updated_sources.append({**src, **metrics})
                
            # Sort sources by their best VGRH score
            updated_sources.sort(key=lambda x: x.get("vgrh_score", 0.0), reverse=True)
            
            # Generate Report
            report_md = report_generator(query, plan_md, updated_sources, verified_evidence)
            
            # Calculate final stats
            elapsed_total = round(asyncio.get_event_loop().time() - pipeline_start, 1)
            stats = {
                "total_sources": len(all_sources),
                "sources_used": sum(1 for s in updated_sources if s.get("vgrh_score", 0.0) > 0.0),
                "total_chunks": len(all_chunks),
                "retrieval_method": "Hybrid (BM25 + Vector RRF)",
                "iterations": iteration,
                "elapsed": elapsed_total
            }
            
            yield log_event(f"Done — report ready in {elapsed_total}s")
            yield complete_step("step-report", "Research report synthesized successfully")
            await asyncio.sleep(0.1)

            # Send final payload to client
            yield f"data: {json.dumps({'type': 'result', 'data': {'report_md': report_md, 'sources': updated_sources, 'evidence': verified_evidence, 'stats': stats}})}\n\n"

        except Exception as e:
            logger.error(f"Pipeline crashed: {e}", exc_info=True)
            yield log_event(f"Pipeline failed: {str(e)}")
            yield f"data: {json.dumps({'type': 'log', 'message': f'Pipeline failed: {str(e)}', 'level': 'error'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# --- Export Endpoints ---

@app.post("/api/export/html")
async def export_html_report(data: Dict[str, Any]):
    """Receives JSON report details and outputs a beautiful, styled self-contained HTML page."""
    query = data.get("query", "Deep Research Report")
    report_md = data.get("report_md", "")
    sources = data.get("sources", [])
    evidence = data.get("evidence", [])
    
    if not report_md:
        raise HTTPException(status_code=400, detail="Missing report markdown content.")
        
    html_content = compile_self_contained_html(query, report_md, sources, evidence)
    
    return Response(
        content=html_content,
        media_type="text/html",
        headers={
            "Content-Disposition": "attachment; filename=Deep_Research_Report.html"
        }
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting uvicorn server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
