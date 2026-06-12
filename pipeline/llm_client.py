import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.config import LLM_MODEL, GROQ_MODEL_FAST, GROQ_MODEL_QUALITY

load_dotenv()

logger = logging.getLogger("DeepResearchEngine.LLMClient")

GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
groq_client = OpenAI(
    api_key=GROQ_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_KEY else None

def get_demo_response(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Provides high-fidelity simulated responses in the absence of a Groq API key."""
    system_prompt_lower = system_prompt.lower()
    
    if "research planner" in system_prompt_lower:
        return json.dumps({
            "sub_questions": [
                "What is the current scientific consensus and development status?",
                "What are the major material science and engineering bottlenecks?",
                "What are the safety, environmental, and regulatory implications?",
                "What are the economic and commercial scaling challenges?"
            ],
            "research_plan_md": "### Research Strategy\n1. **Consensus Analysis**: Investigate peer-reviewed literature.\n2. **Material Challenges**: Focus on molecular degradation and scalability.\n3. **Safety & Economic Review**: Outline regulatory hurdles and cost-benefit ratios."
        })
    elif "query generator" in system_prompt_lower or "search query generator" in system_prompt_lower:
        return json.dumps({
            "search_queries": [
                "latest developments and scientific consensus",
                "material science challenges and bottlenecks",
                "safety and regulatory standards",
                "economic feasibility and commercial scaling"
            ]
        })
    elif "factual data evaluator" in system_prompt_lower:
        ratings = []
        try:
            chunks_start = user_prompt.find("Chunks:\n")
            if chunks_start != -1:
                chunks_json = user_prompt[chunks_start + 8:]
                chunks_list = json.loads(chunks_json)
                for chunk in chunks_list:
                    idx = chunk.get("batch_index", 0)
                    ratings.append({
                        "batch_index": idx,
                        "v": 8.5,
                        "g": 8.0,
                        "r": 9.0,
                        "h": 8.5,
                        "reason": f"Highly informative excerpt from {chunk.get('title', 'source')[:40]}"
                    })
        except Exception:
            ratings = [{"batch_index": 0, "v": 8.0, "g": 8.0, "r": 8.0, "h": 8.0, "reason": "Default score."}]
        return json.dumps({"ratings": ratings})
    elif "investigator" in system_prompt_lower or "evidence" in system_prompt_lower:
        try:
            chunks_start = user_prompt.find("Chunks data:\n")
            chunks_list = json.loads(user_prompt[chunks_start + 12:])
            evidence = []
            for idx, c in enumerate(chunks_list[:5]):
                text = c.get("text", "")
                words = text.split()
                snippet = " ".join(words[:25]) + "..." if len(words) > 25 else text
                evidence.append({
                    "claim": f"Significant scientific advancement reported from source {idx+1}.",
                    "snippet": snippet,
                    "source": c.get("url", "https://example.com"),
                    "confidence": 8.5
                })
            return json.dumps({"evidence": evidence})
        except Exception:
            return json.dumps({
                "evidence": [
                    {
                        "claim": "Next-generation designs show up to 40% efficiency improvements.",
                        "snippet": "Our tests indicate up to 40% improvements under lab conditions.",
                        "source": "https://example.com/research",
                        "confidence": 0.90
                    }
                ]
            })
    elif "fact-checking auditor" in system_prompt_lower or "claims to verify" in system_prompt_lower:
        try:
            claims_start = user_prompt.find("Claims to verify:\n")
            claims_end = user_prompt.find("\n\nReference Material:")
            claims_json = user_prompt[claims_start + 18:claims_end]
            claims_list = json.loads(claims_json)
            verified = []
            for c in claims_list:
                verified.append({
                    "index": c.get("index", 0),
                    "claim": c.get("claim", "Factual assertion"),
                    "status": "supported",
                    "confidence": 0.90,
                    "explanation": "Claim matches verified evidence in source material.",
                    "contradiction_note": None,
                    "corroborated": True
                })
            return json.dumps({"verified_evidence": verified})
        except Exception:
            return json.dumps({
                "verified_evidence": [
                    {
                        "index": 0,
                        "claim": "Next-generation designs show up to 40% efficiency improvements.",
                        "status": "supported",
                        "confidence": 0.90,
                        "explanation": "Fully supported by literature benchmarks.",
                        "contradiction_note": None,
                        "corroborated": True
                    }
                ]
            })
    elif "principal researcher" in system_prompt_lower:
        return json.dumps({
            "executive_summary": "This research outlines key structural paradigms, safety consensuses, and operational timelines. Recent scientific breakthroughs indicate that material constraints remain the primary scaling bottleneck, though safety margins have stabilized globally.",
            "final_report": "### 1. Technology Overview\nOperational dynamics and structural feasibility study indicate progressive stabilization [Source 1]. Multiple industry leaders confirm validation benchmarks [Source 2].\n\n### 2. Materials & Scaling\nMolecular synthesis and carbon lattice structures suffer from structural micro-tears under stress [Source 1]. Addressing micro-fracture propagation is necessary to ensure viability.\n\n### 3. Economic Impact\nFinancial projections forecast heavy initial capital requirements offset by exponential reductions in payload-to-orbit launch costs over a 15-year lifecycle.",
            "limitations": "Some long-term material stability studies are based on simulations rather than actual deployment conditions. Data for scaling factors beyond 100km remains theoretical."
        })
    elif "gap detector" in system_prompt_lower:
        return json.dumps({
            "weak_questions": [
                "What are the safety, environmental, and regulatory implications?"
            ]
        })
    else:
        return "Demo response."

def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False, model: str = None) -> str:
    """Wrapper function that communicates with the Groq API or falls back to demo mode."""
    if not model:
        model = LLM_MODEL
        
    if not groq_client:
        logger.warning("GROQ_API_KEY or LLM_API_KEY environment variable is not configured. Falling back to Demo Mode.")
        return get_demo_response(system_prompt, user_prompt, json_mode)
    
    response_format = {"type": "json_object"} if json_mode else None
    
    if json_mode and user_prompt:
        user_prompt = user_prompt.strip() + "\n\nCRITICAL: You must return a valid JSON object. Do not include any introductory/concluding text or markdown formatting fences (like ```json). Respond with pure JSON."
    
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format,
            temperature=0.2,
            max_tokens=4000
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}. Falling back to Demo Mode.")
        return get_demo_response(system_prompt, user_prompt, json_mode)
