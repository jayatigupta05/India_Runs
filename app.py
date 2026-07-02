"""
Redrob Hackathon — Candidate Ranker Sandbox
Upload a JSON array or JSONL file of candidates → get a ranked CSV back.
"""

import csv
import gzip
import io
import json
import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# JD CONSTANTS
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform Pune Noida India hybrid.
5 to 9 years experience applied ML AI roles product companies not pure services.
Must have: production embeddings retrieval systems sentence-transformers OpenAI embeddings BGE E5 embedding drift index refresh retrieval quality regression.
Must have: vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS operational experience.
Must have: strong Python code quality production.
Must have: evaluation frameworks ranking systems NDCG MRR MAP offline online A/B testing.
Nice to have: LLM fine-tuning LoRA QLoRA PEFT learning to rank XGBoost neural HR-tech recruiting marketplace distributed systems large-scale inference open source contributions AI ML.
Do NOT want: pure research academic no production deployment. LangChain tutorial only without pre-LLM ML production experience. Pure consulting TCS Infosys Wipro Accenture Cognizant Capgemini Hexaware entire career. Computer vision speech robotics without NLP IR. Closed source 5+ years no external validation.
Ideal: 6-8 years total 4-5 applied ML at product companies. Shipped end-to-end ranking search recommendation real users meaningful scale. Located or willing to relocate Noida Pune. Active on platform.
"""

MUST_HAVE_SKILLS = {
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "vector search", "vector database", "vector db", "semantic search",
    "dense retrieval", "hybrid retrieval", "hybrid search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "chroma", "pgvector",
    "information retrieval", "ranking", "learning to rank", "bm25",
    "ndcg", "mrr", "map", "reranking", "re-ranking",
    "nlp", "natural language processing", "transformers", "bert", "llm",
    "large language model", "rag", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "python", "a/b testing", "ab testing", "evaluation framework",
}

NICE_TO_HAVE_SKILLS = {
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "hugging face", "huggingface",
    "mlflow", "weights & biases", "wandb", "fastapi", "flask",
    "docker", "kubernetes", "spark", "airflow", "kafka",
    "recommendation system", "recommendation", "search",
    "distributed systems", "inference optimization", "open source", "github",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hexaware", "mphasis", "tech mahindra",
    "hcl technologies", "hcl tech",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
    "mumbai", "hyderabad", "bangalore", "bengaluru", "chennai",
}

# ---------------------------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------------------------

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def days_since(d):
    if d is None:
        return 9999
    return (date.today() - d).days

def clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))

def normalize(val, lo, hi):
    if hi == lo:
        return 0.5
    return clamp((val - lo) / (hi - lo))

def is_honeypot(c):
    try:
        profile = c["profile"]
        yoe = profile.get("years_of_experience", 0)
        career = c.get("career_history", [])
        skills = c.get("skills", [])
        for role in career:
            dur = role.get("duration_months", 0)
            start = parse_date(role.get("start_date"))
            if start and dur > 0:
                company_age_months = (date.today() - start).days / 30
                if dur > company_age_months + 6:
                    return True
        expert_zero = sum(
            1 for sk in skills
            if sk.get("proficiency") == "expert" and sk.get("duration_months", 1) == 0
        )
        if expert_zero >= 3:
            return True
        total_months = sum(r.get("duration_months", 0) for r in career)
        declared_months = yoe * 12
        if total_months > declared_months * 1.5 + 24:
            return True
        if len(skills) > 20:
            suspicious = sum(
                1 for sk in skills
                if sk.get("proficiency") in ("expert", "advanced")
                and sk.get("endorsements", 0) >= 99
            )
            if suspicious > 15:
                return True
    except Exception:
        pass
    return False

def skill_score(c):
    candidate_skills = set()
    for sk in c.get("skills", []):
        name = sk["name"].lower().strip()
        candidate_skills.add(name)
        for token in name.split():
            candidate_skills.add(token)
    all_text = " ".join([
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
        *[r.get("description", "") for r in c.get("career_history", [])],
    ]).lower()
    assessment_scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    must_hits = sum(1 for sk in MUST_HAVE_SKILLS if sk in candidate_skills or sk in all_text)
    nice_hits = sum(1 for sk in NICE_TO_HAVE_SKILLS if sk in candidate_skills or sk in all_text)
    assessment_bonus = sum(
        (val / 100) * 0.05 for key, val in assessment_scores.items()
        if any(s in key.lower() for s in ["python", "nlp", "ml", "embedding", "retrieval"])
    )
    raw = 0.70 * (must_hits / len(MUST_HAVE_SKILLS)) + \
          0.25 * (nice_hits / len(NICE_TO_HAVE_SKILLS)) + \
          min(0.1, assessment_bonus)
    return clamp(raw)

def career_score(c):
    profile = c["profile"]
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})
    score = 0.0
    yoe = profile.get("years_of_experience", 0)
    if 6 <= yoe <= 8:
        exp_score = 1.0
    elif 4 <= yoe < 6:
        exp_score = 0.7
    elif 8 < yoe <= 12:
        exp_score = 0.8
    elif yoe > 12:
        exp_score = 0.5
    else:
        exp_score = 0.2
    score += 0.20 * exp_score
    title = profile.get("current_title", "").lower()
    ai_titles = {"ml engineer", "machine learning engineer", "ai engineer",
                 "nlp engineer", "research engineer", "applied scientist",
                 "data scientist", "applied ml", "senior engineer",
                 "software engineer", "backend engineer"}
    title_score = 1.0 if any(t in title for t in ai_titles) else \
                  0.5 if any(w in title for w in ["engineer", "scientist", "ml", "ai", "nlp", "data"]) else 0.0
    score += 0.20 * title_score
    all_text = " ".join([c["profile"].get("summary", ""),
                         *[r.get("description", "") for r in career]]).lower()
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(f in r.get("company", "").lower() for f in CONSULTING_FIRMS)
    )
    total_months = sum(r.get("duration_months", 0) for r in career) or 1
    all_consulting = all(
        any(f in r.get("company", "").lower() for f in CONSULTING_FIRMS)
        for r in career
    ) and len(career) >= 2
    company_score = 0.1 if all_consulting else clamp(1.0 - (consulting_months / total_months) * 0.8)
    score += 0.15 * company_score
    prod_keywords = ["production", "deployed", "shipped", "at scale", "real users",
                     "million", "latency", "inference", "serving", "api", "endpoint"]
    prod_score = clamp(sum(1 for kw in prod_keywords if kw in all_text) / 6)
    score += 0.15 * prod_score
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        loc_score = 1.0
    elif country == "india" and relocate:
        loc_score = 0.8
    elif country == "india":
        loc_score = 0.6
    elif relocate:
        loc_score = 0.4
    else:
        loc_score = 0.2
    score += 0.15 * loc_score
    notice = signals.get("notice_period_days", 90)
    notice_score = 1.0 if notice <= 30 else clamp(1.0 - notice / 90) if notice <= 90 else 0.2
    score += 0.10 * notice_score
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min, sal_max = salary.get("min", 0), salary.get("max", 999)
    sal_overlap = clamp((min(sal_max, 80) - max(sal_min, 25)) / 55)
    score += 0.05 * sal_overlap
    return clamp(score)

def behavior_score(c):
    s = c.get("redrob_signals", {})
    score = 0.0
    last_active = parse_date(s.get("last_active_date"))
    days_inactive = days_since(last_active)
    activity = 1.0 if days_inactive <= 14 else \
               0.85 if days_inactive <= 30 else \
               0.6 if days_inactive <= 90 else \
               0.3 if days_inactive <= 180 else 0.05
    score += 0.20 * activity
    if s.get("open_to_work_flag"):
        score += 0.10
    score += 0.15 * s.get("recruiter_response_rate", 0.0)
    score += 0.05 * clamp(1.0 - s.get("avg_response_time_hours", 48) / 48)
    score += 0.10 * (s.get("profile_completeness_score", 0) / 100)
    score += 0.10 * s.get("interview_completion_rate", 0.0)
    gh = s.get("github_activity_score", -1)
    score += 0.10 * (gh / 100 if gh >= 0 else 0.0)
    score += 0.05 * (min(s.get("saved_by_recruiters_30d", 0), 20) / 20)
    verified = (int(s.get("verified_email", False)) +
                int(s.get("verified_phone", False)) +
                int(s.get("linkedin_connected", False))) / 3
    score += 0.10 * verified
    oar = s.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += 0.05 * oar
    return clamp(score)

def build_candidate_text(c):
    parts = [
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
        c["profile"].get("current_title", ""),
        " ".join(sk["name"] for sk in c.get("skills", [])),
        " ".join(r.get("title", "") + " " + r.get("description", "")[:400]
                 for r in c.get("career_history", [])[:4]),
        " ".join(cert.get("name", "") for cert in c.get("certifications", [])),
    ]
    return " ".join(p for p in parts if p).strip()[:2000]

def compute_semantic_scores(candidates):
    texts = [build_candidate_text(c) for c in candidates]
    corpus = [JD_TEXT] + texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                  sublinear_tf=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    return sims

def generate_reasoning(c, rank, sem, skill, career, behav):
    p = c["profile"]
    s = c["redrob_signals"]
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "N/A")
    loc = p.get("location", "N/A")
    notice = s.get("notice_period_days", "?")
    rr = s.get("recruiter_response_rate", 0)
    last_active = s.get("last_active_date", "unknown")
    gh = s.get("github_activity_score", -1)
    jd_relevant = [sk["name"] for sk in c.get("skills", [])
                   if any(m in sk["name"].lower() for m in
                          ["embedding", "vector", "nlp", "retrieval", "python",
                           "transformers", "llm", "rag", "ranking", "faiss",
                           "search", "bert", "fine-tun", "pytorch"])]
    rel_skills_str = ", ".join(jd_relevant[:3]) if jd_relevant else "limited relevant skills"
    concerns = []
    if days_since(parse_date(last_active)) > 90:
        concerns.append(f"inactive {days_since(parse_date(last_active))}d")
    if rr < 0.3:
        concerns.append(f"low response rate ({rr:.0%})")
    if isinstance(notice, int) and notice > 60:
        concerns.append(f"long notice ({notice}d)")
    concern_str = f" Concern: {'; '.join(concerns)}." if concerns else ""
    if rank <= 10:
        tone = f"Strong fit: {title}, {yoe}y exp, skills include {rel_skills_str}."
    elif rank <= 30:
        tone = f"Good fit: {title} with {yoe}y exp; relevant skills: {rel_skills_str}."
    elif rank <= 60:
        tone = f"Moderate fit: {title}, {yoe}y exp; partial skill overlap ({rel_skills_str})."
    else:
        tone = f"Weak fit: {title} with {yoe}y; limited alignment with JD requirements."
    gh_note = f" GitHub score {gh:.0f}/100." if gh > 0 else ""
    return f"{tone}{gh_note} {loc}, notice {notice}d, response rate {rr:.0%}.{concern_str}"

def rank_candidates(candidates, top_n=100):
    honeypot_flags = [is_honeypot(c) for c in candidates]
    skill_scores = np.array([skill_score(c) for c in candidates])
    career_scores = np.array([career_score(c) for c in candidates])
    behavior_scores = np.array([behavior_score(c) for c in candidates])
    fast_score = 0.50 * skill_scores + 0.50 * career_scores
    pre_n = min(5000, len(candidates))
    top_idx = np.argsort(fast_score)[::-1][:pre_n]
    top_candidates = [candidates[i] for i in top_idx]
    sem_scores_top = compute_semantic_scores(top_candidates)
    sem_scores = np.zeros(len(candidates))
    for i, idx in enumerate(top_idx):
        sem_scores[idx] = sem_scores_top[i]
    combined = (0.35 * sem_scores + 0.30 * skill_scores +
                0.20 * career_scores + 0.15 * behavior_scores)
    for i, flag in enumerate(honeypot_flags):
        if flag:
            combined[i] *= 0.05
    actual_top = min(top_n, len(candidates))
    ranked_idx = np.argsort(combined)[::-1][:actual_top]
    results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        c = candidates[idx]
        fs = float(combined[idx])
        results.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": round(fs, 6),
            "reasoning": generate_reasoning(
                c, rank, float(sem_scores[idx]),
                float(skill_scores[idx]), float(career_scores[idx]),
                float(behavior_scores[idx])
            ),
            "_title": c["profile"].get("current_title", ""),
            "_yoe": c["profile"].get("years_of_experience", 0),
            "_location": c["profile"].get("location", ""),
            "_open": c["redrob_signals"].get("open_to_work_flag", False),
            "_notice": c["redrob_signals"].get("notice_period_days", "?"),
            "_github": c["redrob_signals"].get("github_activity_score", -1),
            "_honeypot": honeypot_flags[idx],
            "_sem": float(sem_scores[idx]),
            "_skill": float(skill_scores[idx]),
            "_career": float(career_scores[idx]),
            "_behav": float(behavior_scores[idx]),
        })
    return results, sum(honeypot_flags)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎯 Redrob Candidate Ranker")
st.caption("Narrative Drift-Aware · Multi-Signal · Offline · < 5 min for 100K candidates")

with st.expander("📋 Job Description (what we're ranking against)", expanded=False):
    st.markdown("""
**Role:** Senior AI Engineer — Redrob AI (Series A), Pune / Noida, Hybrid

**Must-have:** Production embeddings & retrieval systems · Vector DBs (FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch) · Python · Evaluation frameworks (NDCG, MRR, MAP) · A/B testing

**Nice-to-have:** LLM fine-tuning (LoRA/QLoRA/PEFT) · Learning to rank · XGBoost · Open source contributions · HR-tech / marketplace experience

**Experience:** 5–9 years, ideally 6–8 · Product companies preferred · Consulting-heavy backgrounds penalised
""")

with st.expander("⚙️ Scoring Architecture", expanded=False):
    st.markdown("""
| Component | Weight | What it measures |
|---|---|---|
| Semantic fit | 35% | TF-IDF bigram cosine similarity, JD vs candidate narrative |
| Skill fit | 30% | Weighted overlap: must-have (70%) + nice-to-have (25%) + assessment bonus (5%) |
| Career fit | 20% | Title, experience band, anti-consulting, production signals, location, notice, salary |
| Behavioral | 15% | Recency, open-to-work, response rate, GitHub, completeness, verifications |

Honeypots (impossible profiles) are detected and pushed below the cutoff with a 0.05× penalty.
""")

st.divider()
st.subheader("Upload Candidates")
st.markdown("Upload a **JSON array** or **JSONL** file (up to 100K candidates). The sample file from your bundle works directly.")

uploaded = st.file_uploader(
    "candidates.json / candidates.jsonl / candidates.jsonl.gz",
    type=["json", "jsonl", "gz"],
    help="JSON array or newline-delimited JSON. Gzipped JSONL also accepted."
)

top_n = st.slider("How many candidates to rank?", min_value=10, max_value=100, value=100, step=10)

if uploaded:
    raw = uploaded.read()
    candidates = []
    try:
        if uploaded.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8")
        # Try JSON array first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                candidates = data
            else:
                st.error("JSON file must be an array of candidate objects.")
        except json.JSONDecodeError:
            # Try JSONL
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        st.error(f"Could not parse file: {e}")

    if candidates:
        st.success(f"Loaded **{len(candidates):,}** candidates.")

        if st.button("🚀 Run Ranker", type="primary", use_container_width=True):
            with st.spinner(f"Ranking {len(candidates):,} candidates across 4 scoring dimensions..."):
                progress = st.progress(0, text="Computing skill + career scores...")
                results, n_honeypots = rank_candidates(candidates, top_n=top_n)
                progress.progress(100, text="Done!")

            st.success(f"✅ Ranked **{len(results)}** candidates · **{n_honeypots}** honeypots detected & penalised")

            # --- Score breakdown chart ---
            df = pd.DataFrame(results)
            display_df = df[["rank", "candidate_id", "_title", "_yoe", "_location",
                              "_open", "_notice", "_github", "score",
                              "_sem", "_skill", "_career", "_behav", "_honeypot", "reasoning"]].copy()
            display_df.columns = ["Rank", "Candidate ID", "Title", "YoE", "Location",
                                   "Open", "Notice(d)", "GitHub", "Score",
                                   "Semantic", "Skill", "Career", "Behavior", "Honeypot?", "Reasoning"]

            st.subheader(f"Top {len(results)} Candidates")
            st.dataframe(
                display_df.style.background_gradient(subset=["Score"], cmap="Greens")
                                .format({"Score": "{:.4f}", "Semantic": "{:.3f}",
                                         "Skill": "{:.3f}", "Career": "{:.3f}",
                                         "Behavior": "{:.3f}", "YoE": "{:.1f}"}),
                use_container_width=True,
                height=500,
            )

            # --- Score distribution ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Top Score", f"{results[0]['score']:.4f}")
            col2.metric("Rank #10 Score", f"{results[min(9,len(results)-1)]['score']:.4f}")
            col3.metric("Rank #50 Score", f"{results[min(49,len(results)-1)]['score']:.4f}")
            col4.metric("Honeypots Flagged", n_honeypots)

            # --- Download CSV ---
            csv_rows = [{"candidate_id": r["candidate_id"], "rank": r["rank"],
                          "score": r["score"], "reasoning": r["reasoning"]}
                         for r in results]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            writer.writerows(csv_rows)
            csv_str = buf.getvalue()

            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_str,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            st.caption("Rename the downloaded file to your team's registered participant ID before uploading to the portal.")
