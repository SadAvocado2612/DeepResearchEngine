import json
import logging
from typing import List, Dict, Any
from pipeline.llm_client import call_llm
from pipeline.config import GROQ_MODEL_FAST, GROQ_MODEL_QUALITY

logger = logging.getLogger("DeepResearchEngine.Planner")

def research_planner(query: str) -> tuple[List[str], str]:
    """Decomposes the primary research topic into detailed sub-questions.
    
    Inputs:
        query (str): The primary user research query.
        
    Outputs:
        tuple[List[str], str]: (sub_questions, research_plan_md).
    """
    logger.info("Executing research_planner...")
    system_prompt = (
        "You are an expert research planner. Decompose the user's research topic into "
        "3 to 5 logical, high-impact sub-questions that must be answered to form a complete overview. "
        "Additionally, provide a detailed Markdown description of the research plan.\n\n"
        "CRITICAL: You must return your response as a valid JSON object. Do not include any explanation outside the JSON. "
        "Do not wrap the response in markdown code blocks. The JSON object must contain exactly two keys:\n"
        '- "sub_questions": a list of strings representing the sub-questions\n'
        '- "research_plan_md": a markdown text block detailing the strategy\n\n'
        'Example JSON output format:\n'
        '{\n'
        '  "sub_questions": ["question 1", "question 2"],\n'
        '  "research_plan_md": "Markdown strategy description"\n'
        '}'
    )
    user_prompt = f"Research Topic: {query}"
    
    try:
        result = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_QUALITY)
        parsed = json.loads(result)
        sub_questions = parsed.get("sub_questions", [])
        research_plan_md = parsed.get("research_plan_md", "")
        return sub_questions, research_plan_md
    except Exception as e:
        logger.error(f"Research planner failed: {e}")
        return [
            "What is the background and consensus?",
            "What are the primary technical specifications?",
            "What are the key safety and scaling challenges?"
        ], "### Default Research Plan\nGenerated due to plan parse error."

def query_generator(sub_questions: List[str]) -> List[str]:
    """Generates optimized search engine keywords and query strings.
    
    Inputs:
        sub_questions (List[str]): List of research sub-questions.
        
    Outputs:
        List[str]: Search engine query strings.
    """
    logger.info("Executing query_generator...")
    system_prompt = (
        "You are an expert search engine query generator. Given a list of research sub-questions, "
        "generate a list of search queries optimized for Tavily or DuckDuckGo. "
        "Generate exactly 1 to 2 highly descriptive queries per sub-question. "
        "Return the output as a JSON object with a single key 'search_queries' containing the list of query strings."
    )
    user_prompt = f"Sub-questions:\n" + "\n".join(f"- {q}" for q in sub_questions)
    
    try:
        result = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_FAST)
        parsed = json.loads(result)
        raw_queries = parsed.get("search_queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = [raw_queries]
        queries = []
        for item in raw_queries:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        queries.append(v)
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, str):
                                queries.append(x)
            elif isinstance(item, str):
                queries.append(item)
            else:
                queries.append(str(item))
        return queries
    except Exception as e:
        logger.error(f"Query generator failed: {e}")
        return [q[:50] for q in sub_questions]

def gap_detector(query: str, sub_questions: List[str], evidence: List[Dict[str, Any]]) -> List[str]:
    """Identifies which research sub-questions lack sufficient backing evidence.
    
    Inputs:
        query (str): The primary user query.
        sub_questions (List[str]): The original research sub-questions.
        evidence (List[Dict]): Currently extracted evidence list.
        
    Outputs:
        List[str]: Sub-questions requiring secondary research due to gaps.
    """
    logger.info("Executing gap_detector...")
    if not evidence:
        logger.info("No evidence extracted yet. All sub-questions flagged as gaps.")
        return sub_questions
        
    system_prompt = (
        "You are a senior research auditor. Review the primary research query, the proposed sub-questions, "
        "and the currently extracted verified evidence items. Identify which of the sub-questions "
        "have weak, missing, or low-confidence evidence. "
        "You must return your evaluation in JSON format containing a list of weak sub-questions. "
        "The JSON object must have a single key 'weak_questions' mapping to a list of sub-questions."
    )
    
    evidence_summary = []
    for idx, ev in enumerate(evidence):
        evidence_summary.append({
            "index": idx + 1,
            "claim": ev.get("claim"),
            "confidence": ev.get("confidence"),
            "source": ev.get("source"),
            "status": ev.get("status")
        })
        
    user_prompt = (
        f"Primary Research Query: {query}\n\n"
        f"Proposed Sub-questions:\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n\n"
        f"Current Extracted Evidence:\n{json.dumps(evidence_summary, indent=2)}"
    )
    
    try:
        result = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_FAST)
        parsed = json.loads(result)
        raw_weak = parsed.get("weak_questions", [])
        if not isinstance(raw_weak, list):
            raw_weak = [raw_weak]
        weak_questions = []
        for item in raw_weak:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        weak_questions.append(v)
            elif isinstance(item, str):
                weak_questions.append(item)
            else:
                weak_questions.append(str(item))
        
        # Verify that the returned sub-questions are actually from the original sub-questions list
        validated_gaps = []
        for wq in weak_questions:
            # Match against sub_questions using case-insensitive check
            matched = False
            for sq in sub_questions:
                if wq.lower().strip("?. ") in sq.lower():
                    validated_gaps.append(sq)
                    matched = True
                    break
            if not matched and wq in sub_questions:
                validated_gaps.append(wq)
                
        # Remove duplicates
        validated_gaps = list(set(validated_gaps))
        logger.info(f"Gap detector flagged {len(validated_gaps)} sub-questions: {validated_gaps}")
        return validated_gaps
    except Exception as e:
        logger.error(f"Gap detector failed: {e}")
        return []
