import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI

# ==============================
# CONFIG – SET YOUR KEY HERE
# ==============================
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# ================
# FILE PATHS
# ================
# Adjust these if your files are in a different folder
QUESTIONS_JSON_PATH = "./output/geography_questions.json"
SYLLABUS_TEXT_PATH = "./output/syllabus_text.txt"

RESULTS_JSON_PATH = "./output/geography_llm_alignment_results.json"
REPORT_MD_PATH = "./output/geography_llm_alignment_report.md"

# ================
# MODEL CONFIG
# ================
OPENAI_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 900         # characters per chunk
CHUNK_OVERLAP = 150      # overlap between chunks
TOP_K = 5                # number of syllabus chunks retrieved per question


@dataclass
class QuestionAnalysis:
    question_number: int
    question_text: str
    matched_clusters: List[str]
    matched_topics: List[str]
    assessment_objectives: List[str]
    alignment_score: float
    alignment_strength: str
    alignment_explanation: str
    retrieved_chunks: List[Dict[str, Any]]


def load_questions(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_syllabus_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Syllabus text not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple char-based chunking with overlap.
    Good enough for POC, avoids tokenization complexity.
    """
    text = text.replace("\r", "")
    text = text.strip()
    chunks = []

    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using OpenAI embeddings.
    Returns an array of shape (N, D).
    """
    batch_size = 64
    all_embs: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding syllabus chunks"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        for d in resp.data:
            all_embs.append(np.array(d.embedding, dtype="float32"))

    return np.vstack(all_embs)


def embed_single(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a cosine-similarity FAISS index (L2-normalised inner product).
    """
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


# ==============================
# PROMPTING
# ==============================
def build_alignment_prompt(
    question_text: str,
    syllabus_chunks: List[Tuple[int, str]]
) -> List[Dict[str, str]]:
    """
    Build messages for GPT-4o based on retrieved syllabus chunks and question text.
    """
    syllabus_context = ""
    for cid, ctext in syllabus_chunks:
        syllabus_context += f"\n--- SYLLABUS CHUNK {cid} ---\n{ctext}\n"

    system_msg = """
You are a senior Geography curriculum and assessment specialist.

Your task is to evaluate how well an O-Level Geography exam question aligns with
the official Upper Secondary Geography syllabus (Syllabus 2279).

You are given:
- A single exam question (which may contain multiple sub-parts).
- Several retrieved syllabus chunks.

You must:
1. Identify which syllabus CLUSTER(S) the question belongs to, e.g.
   - "Cluster 1: Geography in Everyday Life"
   - "Cluster 2: Tourism"
   - "Cluster 3: Climate"
   - "Cluster 4: Tectonics"
   - "Cluster 5: Singapore"

2. Identify which specific TOPIC(S) the question relates to, e.g.,
   - "Cluster 2 Topic 2.1: What is tourism and impacts of tourism?"
   - "Cluster 1 Topic 1.3: Geographical Methods"

3. Identify which Assessment Objectives (AOs) are mainly tested:
   - AO1: Knowledge & Understanding
   - AO2: Skills & Application
   - AO3: Evaluation & Judgement

4. Provide an alignment score on a 0–5 scale:
   - 0 = No alignment
   - 1 = Very weak / vague thematic link
   - 2 = Partial but incomplete alignment
   - 3 = Reasonable alignment
   - 4 = Strong alignment
   - 5 = Very strong, explicit alignment to clearly stated syllabus content

5. Provide an alignment_strength label:
   - "None", "Weak", "Moderate", or "Strong"

6. Give a concise explanation citing which syllabus ideas/output the question is testing.

You MUST base your reasoning ONLY on the syllabus text in the provided chunks.
Return STRICT JSON and nothing else.
"""

    user_msg = f"""
EXAM QUESTION:
{question_text}

RELEVANT SYLLABUS CONTEXT:
{syllabus_context}

Now perform the evaluation and return JSON in this exact structure:
{{
  "matched_clusters": ["..."],
  "matched_topics": ["..."],
  "assessment_objectives": ["AO1", "AO2", "..."],
  "alignment_score": 0-5,
  "alignment_strength": "None|Weak|Moderate|Strong",
  "alignment_explanation": "short explanation here"
}}
"""

    return [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()},
    ]


def call_gpt_for_alignment(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.1,
    )
    content = resp.choices[0].message.content.strip()

    # Handle ```json fenced code blocks if present
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()

    try:
        data = json.loads(content)
    except Exception:
        # Fallback if parsing fails
        data = {
            "matched_clusters": [],
            "matched_topics": [],
            "assessment_objectives": [],
            "alignment_score": 0,
            "alignment_strength": "None",
            "alignment_explanation": f"Failed to parse LLM JSON. Raw output: {content[:400]}",
        }
    return data


# ==============================
# REPORT GENERATION
# ==============================
def generate_markdown_report(
    analyses: List[QuestionAnalysis],
    summary: Dict[str, Any]
) -> str:
    lines = []
    lines.append("# Geography LLM Alignment & Topic Weightage Report\n")
    lines.append("## 1. Per-Question Alignment Overview\n")

    for qa in analyses:
        lines.append(f"### Question {qa.question_number}\n")
        # Show first ~200 chars of the question for readability
        short_q = qa.question_text.replace("\n", " ")
        if len(short_q) > 220:
            short_q = short_q[:220] + "..."
        lines.append(f"**Question (snippet):** {short_q}\n")
        lines.append(f"- **Matched Clusters:** {', '.join(qa.matched_clusters) or '—'}")
        lines.append(f"- **Matched Topics:** {', '.join(qa.matched_topics) or '—'}")
        lines.append(f"- **Assessment Objectives (AOs):** {', '.join(qa.assessment_objectives) or '—'}")
        lines.append(f"- **Alignment Score:** {qa.alignment_score} ({qa.alignment_strength})")
        lines.append(f"- **Explanation:** {qa.alignment_explanation}\n")

    lines.append("\n---\n")
    lines.append("## 2. Topic & Cluster Weightage Summary\n")

    total_questions = summary.get("total_questions", 0)
    lines.append(f"Total questions analysed: **{total_questions}**\n")

    lines.append("### 2.1 Cluster Weightage\n")
    lines.append("| Cluster | Count | Proportion |")
    lines.append("|---------|-------|------------|")
    for c, info in summary.get("cluster_weightage", {}).items():
        lines.append(f"| {c} | {info['count']} | {info['proportion']:.2f} |")

    lines.append("\n### 2.2 Topic Weightage\n")
    lines.append("| Topic | Count | Proportion |")
    lines.append("|-------|-------|------------|")
    for t, info in summary.get("topic_weightage", {}).items():
        lines.append(f"| {t} | {info['count']} | {info['proportion']:.2f} |")

    lines.append("\n### 2.3 Assessment Objectives Distribution\n")
    lines.append("| AO | Count |")
    lines.append("|----|-------|")
    for ao, cnt in summary.get("assessment_objectives_distribution", {}).items():
        lines.append(f"| {ao} | {cnt} |")

    lines.append("\n---\n")
    lines.append("## 3. Interpretation & Next Steps (Example Talking Points)\n")
    lines.append("- Check if Cluster coverage (e.g., Everyday Life, Tourism, Climate, Tectonics, Singapore) ")
    lines.append("  matches intended syllabus emphasis.\n")
    lines.append("- Check if AO2/AO3 (skills & evaluation) are reasonably represented relative to AO1.\n")
    lines.append("- Identify any under-represented topics or clusters which may need more questions in future papers.\n")

    return "\n".join(lines)


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    os.makedirs("./output", exist_ok=True)

    print("Loading exam questions...")
    questions = load_questions(QUESTIONS_JSON_PATH)
    print(f"  Loaded {len(questions)} questions")

    print("Loading syllabus text...")
    syllabus_text = load_syllabus_text(SYLLABUS_TEXT_PATH)
    print(f"  Syllabus text length: {len(syllabus_text)} characters")

    print("Chunking syllabus text...")
    syllabus_chunks = chunk_text(syllabus_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Total syllabus chunks: {len(syllabus_chunks)}")

    if not syllabus_chunks:
        raise ValueError("No syllabus chunks produced; check syllabus_text.txt content.")

    print("Embedding syllabus chunks & building FAISS index...")
    syllabus_embs = embed_texts(syllabus_chunks)
    index = build_faiss_index(syllabus_embs)

    all_results: List[QuestionAnalysis] = []

    print("\nRunning GPT-based alignment & topic evaluation per question...\n")
    for q in tqdm(questions, desc="Questions"):
        q_num = q.get("question_number")
        q_text = q.get("text", "")

        # Embed question
        q_emb = embed_single(q_text).reshape(1, -1)

        # Retrieve top-K syllabus chunks
        scores, idxs = index.search(q_emb, TOP_K)
        idxs = idxs[0]
        scores = scores[0]

        retrieved = []
        for rank, (sid, score) in enumerate(zip(idxs, scores), start=1):
            if sid < 0 or sid >= len(syllabus_chunks):
                continue
            retrieved.append({
                "rank": rank,
                "chunk_id": int(sid),
                "similarity": float(score),
                "text": syllabus_chunks[sid]
            })

        # Build GPT prompt using retrieved chunks
        messages = build_alignment_prompt(
            question_text=q_text,
            syllabus_chunks=[(r["chunk_id"], r["text"]) for r in retrieved]
        )

        llm_result = call_gpt_for_alignment(messages)

        qa = QuestionAnalysis(
            question_number=q_num,
            question_text=q_text,
            matched_clusters=llm_result.get("matched_clusters", []),
            matched_topics=llm_result.get("matched_topics", []),
            assessment_objectives=llm_result.get("assessment_objectives", []),
            alignment_score=float(llm_result.get("alignment_score", 0)),
            alignment_strength=llm_result.get("alignment_strength", "None"),
            alignment_explanation=llm_result.get("alignment_explanation", ""),
            retrieved_chunks=retrieved,
        )
        all_results.append(qa)

    # ==============================
    # AGGREGATE WEIGHTAGE SUMMARY
    # ==============================
    topic_counts: Dict[str, int] = {}
    cluster_counts: Dict[str, int] = {}
    ao_counts: Dict[str, int] = {}

    for qa in all_results:
        for t in qa.matched_topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        for c in qa.matched_clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        for ao in qa.assessment_objectives:
            ao_counts[ao] = ao_counts.get(ao, 0) + 1

    total_questions = len(all_results)
    cluster_weightage = {
        c: {
            "count": cnt,
            "proportion": cnt / total_questions if total_questions > 0 else 0.0,
        }
        for c, cnt in cluster_counts.items()
    }
    topic_weightage = {
        t: {
            "count": cnt,
            "proportion": cnt / total_questions if total_questions > 0 else 0.0,
        }
        for t, cnt in topic_counts.items()
    }

    summary = {
        "total_questions": total_questions,
        "cluster_weightage": cluster_weightage,
        "topic_weightage": topic_weightage,
        "assessment_objectives_distribution": ao_counts,
    }

    # ==============================
    # SAVE JSON RESULTS
    # ==============================
    print(f"\n▶ Saving JSON results to {RESULTS_JSON_PATH} ...")
    results_out = {
        "per_question": [asdict(qa) for qa in all_results],
        "summary": summary,
    }
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)

    # ==============================
    # SAVE MARKDOWN REPORT
    # ==============================
    print(f"▶ Saving Markdown report to {REPORT_MD_PATH} ...")
    report_md = generate_markdown_report(all_results, summary)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n✅ Done.")
    print(f"- Detailed JSON: {RESULTS_JSON_PATH}")
    print(f"- Human-readable report: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
