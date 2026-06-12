import datetime
import logging
import re
import json
from typing import List, Dict, Any
from pipeline.llm_client import call_llm
from pipeline.config import GROQ_MODEL_QUALITY

logger = logging.getLogger("DeepResearchEngine.Reporter")



def markdown_to_html_simple(md_text: str) -> str:
    """Performs regex-based basic translation of Markdown to HTML for self-contained exports."""
    html = md_text
    
    # Headers
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold / Italic
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Inline code
    html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
    
    # Blockquotes
    html = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Unordered Lists
    # Simple list matching
    html = re.sub(r'^\s*-\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    # Wrap loose <li> groups in <ul>
    # We do a basic line splitter for paragraphs
    lines = html.split('\n')
    in_list = False
    for idx, line in enumerate(lines):
        if line.strip().startswith('<li>'):
            if not in_list:
                lines[idx] = '<ul>\n' + line
                in_list = True
        else:
            if in_list:
                lines[idx - 1] = lines[idx - 1] + '\n</ul>'
                in_list = False
    html = '\n'.join(lines)
    
    # Paragraphs (loose lines wrapped in p, except tags)
    paragraphs = []
    for block in html.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('<h') or block.startswith('<ul') or block.startswith('<li') or block.startswith('<block') or block.startswith('|') or block.startswith('<table'):
            paragraphs.append(block)
        else:
            paragraphs.append(f"<p>{block}</p>")
            
    return '\n\n'.join(paragraphs)

def report_generator(query: str, plan_md: str, sources: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> str:
    """Synthesizes the final cited Markdown report.
    
    Automatically collects uncertain/contradicted claims to compile the Limitations section.
    
    Inputs:
        query (str): Research query.
        plan_md (str): Detailed research plan.
        sources (List[Dict]): Discovered and ranked sources.
        evidence (List[Dict]): Extracted and verified claims.
        
    Outputs:
        str: Cited Markdown report.
    """
    logger.info("Executing report_generator...")
    
    # Pre-structure mapping of sources for LLM citations
    source_mapping = []
    for idx, src in enumerate(sources):
        source_mapping.append(f"[Source {idx+1}]: URL: {src['url']} | Title: {src['title']}")
        
    evidence_str = json.dumps(evidence, indent=2)
    source_mapping_str = "\n".join(source_mapping)
    
    system_prompt = (
        "You are a principal researcher. Generate a comprehensive, professional research synthesis report in Markdown. "
        "The report must contain detailed prose answering the user query. "
        "You MUST insert inline citations in the format '[Source N]' (where N is the 1-based index of the source) "
        "wherever you reference factual evidence. You are given a mapping of sources and verified evidence.\n\n"
        "The response must provide two sections in Markdown:\n"
        "1. Executive Summary (a short direct answer summarizing key findings)\n"
        "2. Final Report (a structured detailed analysis with inline [citations])\n"
        "3. Limitations (unsupported claims, data gaps, or uncertainties detected)\n\n"
        "Make sure that these sections are properly formatted and readable. \n"
        "Format the output JSON as: {'executive_summary': '...', 'final_report': '...', 'limitations': '...'}"
    )
    
    user_prompt = (
        f"Topic: {query}\n\n"
        f"Verified Evidence:\n{evidence_str}\n\n"
        f"Source Mapping:\n{source_mapping_str}"
    )
    
    try:
        response_str = call_llm(system_prompt, user_prompt, json_mode=True, model=GROQ_MODEL_QUALITY)
        response_json = json.loads(response_str)
        exec_summary = response_json.get("executive_summary", "Synthesis pending.")
        final_report_body = response_json.get("final_report", "Report body pending.")
        limitations_narrative = response_json.get("limitations", "No limitations specified.")
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        exec_summary = f"Summary of findings for query: {query}"
        final_report_body = "Error generating narrative. Review the source and evidence tables."
        limitations_narrative = "Could not verify limitations."

    # Compile dynamic limitations from claims audit
    limitations_list = []
    for ev in evidence:
        status = ev.get("status", "").lower()
        if status in ["uncertain", "contradicted"]:
            prefix = "🔶 [Uncertain]" if status == "uncertain" else "⚠️ [Contradicted]"
            limitations_list.append(
                f"- {prefix} **{ev['claim']}**: {ev.get('explanation', 'Evidence verification flagged.')}"
            )
            
    limitations_section = limitations_narrative
    if limitations_list:
        limitations_section += "\n\n### Automatically Flagged Claims & Uncertainties\n" + "\n".join(limitations_list)

    # Construct the Markdown tables manually to guarantee correct formatting
    # Source Table
    source_table_md = "| Source | Type | VGRH Score | Tier | Retrieval | Iteration | Reason Selected |\n|---|---|---|---|---|---|---|\n"
    for idx, src in enumerate(sources):
        vgrh = src.get("vgrh_score", "N/A")
        v = src.get("v", "N/A")
        g = src.get("g", "N/A")
        r = src.get("r", "N/A")
        h = src.get("h", "N/A")
        tier = src.get("authority_tier", "🌐 General Web")
        ret_method = src.get("retrieval_method", "BM25")
        iter_run = src.get("iteration", 1)
        reason = src.get("reason", "Relevant discovered source.")
        title = src.get("title", f"Source {idx+1}")
        
        source_table_md += f"| [Source {idx+1}]({src['url']})<br>*{title}* | {src['type'].upper()} | {vgrh} <br>*(V:{v} G:{g} R:{r} H:{h})* | {tier} | {ret_method} | Iter {iter_run} | {reason} |\n"
        
    # Evidence Table
    evidence_table_md = "| Claim | Evidence Snippet | Source | Confidence | Status |\n|---|---|---|---|---|\n"
    for ev in evidence:
        claim = ev.get("claim", "N/A")
        snippet = ev.get("snippet", "N/A")
        source = ev.get("source", "N/A")
        confidence = ev.get("confidence", "N/A")
        status = ev.get("status", "N/A").upper()
        
        source_label = source
        for idx, src in enumerate(sources):
            if src["url"] == source:
                source_label = f"[Source {idx+1}]({source})"
                break
                
        # Status icon
        status_icon = "✅ SUPPORTED"
        if status.lower() == "uncertain":
            status_icon = "🔶 UNCERTAIN"
        elif status.lower() == "contradicted":
            status_icon = "⚠️ CONTRADICTED"
            
        evidence_table_md += f"| {claim} | *\"{snippet}\"* | {source_label} | {confidence} | {status_icon} |\n"

    report_md = f"""# Deep Research Report: {query}

## Executive Summary
{exec_summary}

## Research Plan
{plan_md}

## Source Table
{source_table_md}

## Evidence Table
{evidence_table_md}

## Final Report
{final_report_body}

## Limitations
{limitations_section}
"""
    return report_md

def compile_self_contained_html(query: str, report_md: str, sources: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> str:
    """Compiles a single, styled self-contained HTML page representing the report.
    
    Inputs:
        query (str): Original query topic.
        report_md (str): Formatted Markdown report.
        sources (List[Dict]): Sources list.
        evidence (List[Dict]): Extracted and verified evidence.
        
    Outputs:
        str: Self-contained HTML document.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate stats
    total_claims = len(evidence)
    uncertain_claims = sum(1 for e in evidence if e["status"] == "uncertain")
    contradicted_claims = sum(1 for e in evidence if e["status"] == "contradicted")
    supported_claims = sum(1 for e in evidence if e["status"] == "supported")
    avg_conf = round(sum(e["confidence"] for e in evidence) / total_claims, 2) if total_claims > 0 else 0.0
    
    # Translate MD body to HTML
    body_html = markdown_to_html_simple(report_md)
    
    # Generate Sources list
    sources_rows = ""
    for idx, src in enumerate(sources):
        vgrh = src.get("vgrh_score", "N/A")
        v = src.get("v", "N/A")
        g = src.get("g", "N/A")
        r = src.get("r", "N/A")
        h = src.get("h", "N/A")
        tier = src.get("authority_tier", "🌐 General Web")
        ret_method = src.get("retrieval_method", "BM25")
        iter_run = src.get("iteration", 1)
        
        sources_rows += f"""
        <tr>
            <td><strong>[Source {idx+1}]</strong><br><a href="{src['url']}" target="_blank">{src['url']}</a><br><small>{src.get('title', '')}</small></td>
            <td>{src['type'].upper()}</td>
            <td><strong>{vgrh}</strong><br><small>(V:{v} G:{g} R:{r} H:{h})</small></td>
            <td>{tier}</td>
            <td>{ret_method}</td>
            <td>Iter {iter_run}</td>
        </tr>
        """
        
    # Generate Evidence cards HTML
    evidence_cards = ""
    for ev in evidence:
        claim = ev.get("claim", "N/A")
        snippet = ev.get("snippet", "N/A")
        source = ev.get("source", "N/A")
        confidence = ev.get("confidence", 0.0)
        status = ev.get("status", "supported").lower()
        explanation = ev.get("explanation", "")
        context = ev.get("chunk_context", snippet)
        
        conf_pct = int(confidence * 100)
        conf_color = "#10b981" # green
        if confidence < 0.5:
            conf_color = "#ef4444" # red
        elif confidence < 0.75:
            conf_color = "#f59e0b" # orange
            
        badge_class = "supported"
        badge_icon = "✅ Supported"
        if status == "uncertain":
            badge_class = "uncertain"
            badge_icon = "🔶 Uncertain"
        elif status == "contradicted":
            badge_class = "contradicted"
            badge_icon = "⚠️ Contradicted"
            
        evidence_cards += f"""
        <div class="evidence-card {badge_class}">
            <div class="card-header">
                <strong>{claim}</strong>
                <span class="badge {badge_class}">{badge_icon}</span>
            </div>
            <blockquote>"{snippet}"</blockquote>
            <div class="source-link">Source: <a href="{source}" target="_blank">{source}</a></div>
            <div class="confidence-bar-container">
                <small>Confidence: {confidence}</small>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width: {conf_pct}%; background-color: {conf_color};"></div>
                </div>
            </div>
            <div class="explanation-text"><small>{explanation}</small></div>
            <details>
                <summary>View Context</summary>
                <div class="context-block">{context}</div>
            </details>
        </div>
        """

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Deep Research Export</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #1a202c;
            background-color: #ffffff;
            margin: 0;
            padding: 3rem;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #1a202c;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }}
        h1 {{ font-size: 2.25rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; border-bottom: 1px solid #edf2f7; padding-bottom: 0.25rem; }}
        h3 {{ font-size: 1.25rem; }}
        
        .header-meta {{
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
            font-family: Arial, sans-serif;
            color: #718096;
            font-size: 0.9rem;
        }}
        
        .banner {{
            background-color: #f7fafc;
            border-left: 4px solid #4a5568;
            padding: 1rem;
            margin-bottom: 2rem;
            font-family: Arial, sans-serif;
            border-radius: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-family: Arial, sans-serif;
            font-size: 0.85rem;
        }}
        th {{
            background-color: #edf2f7;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #cbd5e0;
        }}
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }}
        
        blockquote {{
            border-left: 4px solid #cbd5e0;
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            font-style: italic;
            background-color: #f7fafc;
        }}
        
        code {{
            font-family: monospace;
            background-color: #f7fafc;
            padding: 2px 4px;
            font-size: 0.9em;
            border-radius: 3px;
        }}
        
        .evidence-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
            font-family: Arial, sans-serif;
        }}
        
        .evidence-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .evidence-card.supported {{ border-left: 4px solid #10b981; }}
        .evidence-card.uncertain {{ border-left: 4px solid #f59e0b; }}
        .evidence-card.contradicted {{ border-left: 4px solid #ef4444; }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .badge.supported {{ background-color: #d1fae5; color: #065f46; }}
        .badge.uncertain {{ background-color: #fef3c7; color: #92400e; }}
        .badge.contradicted {{ background-color: #fee2e2; color: #991b1b; }}
        
        .source-link a {{
            color: #3182ce;
            text-decoration: none;
            word-break: break-all;
        }}
        .source-link a:hover {{ text-decoration: underline; }}
        
        .confidence-bar-container {{
            margin: 0.75rem 0;
        }}
        .confidence-track {{
            height: 6px;
            background-color: #edf2f7;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 3px;
        }}
        .confidence-fill {{
            height: 100%;
        }}
        
        .explanation-text {{
            color: #4a5568;
            margin-bottom: 0.5rem;
        }}
        
        details {{
            margin-top: 0.5rem;
            cursor: pointer;
        }}
        summary {{
            font-size: 0.8rem;
            color: #4a5568;
            font-weight: bold;
        }}
        .context-block {{
            background-color: #f7fafc;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-top: 4px;
            color: #4a5568;
        }}
        
        @media print {{
            body {{
                padding: 1.5cm;
            }}
            .evidence-card {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <h1>Deep Research Report</h1>
    <div class="header-meta">
        <strong>Original Query:</strong> "{query}"<br>
        <strong>Generated Timestamp:</strong> {timestamp}
    </div>
    
    <div class="banner">
        <strong>Confidence Summary Banner:</strong><br>
        {supported_claims} claims supported &middot; {uncertain_claims} uncertain &middot; {contradicted_claims} contradicted &middot; Avg confidence: {avg_conf}
    </div>
    
    <div class="report-body">
        {body_html}
    </div>
    
    <h2>Sources Table</h2>
    <table>
        <thead>
            <tr>
                <th>Source URL / Title</th>
                <th>Type</th>
                <th>VGRH Score</th>
                <th>Tier</th>
                <th>Retrieval Method</th>
                <th>Iteration</th>
            </tr>
        </thead>
        <tbody>
            {sources_rows}
        </tbody>
    </table>
    
    <h2>Evidence Details</h2>
    <div class="evidence-grid">
        {evidence_cards}
    </div>
</body>
</html>
"""
    return html_template


