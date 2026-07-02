#!/usr/bin/env python3
"""
Redrob Hackathon — Narrative Drift-Aware Candidate Ranker
==========================================================
Architecture:
  1. Semantic fit   — sentence-transformer embeddings of JD vs candidate narrative
  2. Skill fit      — weighted overlap of JD must-have/nice-to-have skills
  3. Career fit     — title trajectory, company type, experience band
  4. Honeypot guard — flag impossible profiles before ranking
  5. Behavioral modifier — redrob_signals as an availability/engagement multiplier

Final score = (0.35*semantic + 0.30*skill + 0.20*career + 0.15*behavior) * honeypot_mask
"""

import argparse
import csv
import gzip
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# JD CONSTANTS  (derived from job_description.md)
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

# Must-have skills (high weight)
MUST_HAVE_SKILLS = {
    # embeddings / retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "vector search", "vector database", "vector db", "semantic search",
    "dense retrieval", "hybrid retrieval", "hybrid search",
    # specific vector dbs
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "chroma", "pgvector",
    # ranking / retrieval
    "information retrieval", "ranking", "learning to rank", "bm25",
    "ndcg", "mrr", "map", "reranking", "re-ranking",
    # LLMs / NLP
    "nlp", "natural language processing", "transformers", "bert", "llm",
    "large language model", "rag", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    # python
    "python",
    # evaluation
    "a/b testing", "ab testing", "evaluation framework", "offline evaluation",
}

NICE_TO_HAVE_SKILLS = {
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "hugging face", "huggingface",
    "mlflow", "weights & biases", "wandb",
    "fastapi", "flask", "docker", "kubernetes",
    "spark", "airflow", "kafka",
    "recommendation system", "recommendation", "search",
    "distributed systems", "inference optimization",
    "open source", "github",
    "product company", "startup",
}

# Disqualifying signals
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hexaware", "mphasis", "tech mahindra",
    "hcl technologies", "hcl tech",
}

# Experience band: 5-9 years preferred, 6-8 ideal
EXP_MIN, EXP_IDEAL_MIN, EXP_IDEAL_MAX, EXP_MAX = 4, 6, 8, 12

# Notice period preference: < 30 days ideal, < 60 acceptable
NOTICE_IDEAL = 30
NOTICE_OK = 90

# Salary range (INR LPA) — role is senior, likely 30-60 LPA
SALARY_JD_MIN, SALARY_JD_MAX = 25, 80

# Preferred locations
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
    "mumbai", "hyderabad", "bangalore", "bengaluru", "chennai",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d: date) -> int:
    if d is None:
        return 9999
    return (date.today() - d).days


def normalize(val, lo, hi):
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# HONEYPOT DETECTION
# ---------------------------------------------------------------------------

def is_honeypot(c: dict) -> bool:
    """Return True if profile is internally inconsistent — likely a honeypot."""
    try:
        profile = c["profile"]
        yoe = profile.get("years_of_experience", 0)
        career = c.get("career_history", [])
        skills = c.get("skills", [])

        # Check 1: experience at company that's younger than their tenure
        for role in career:
            dur = role.get("duration_months", 0)
            start = parse_date(role.get("start_date"))
            if start and dur > 0:
                company_age_months = (date.today() - start).days / 30
                if dur > company_age_months + 6:  # 6 month buffer
                    return True

        # Check 2: skill claimed as expert with 0 months duration
        expert_zero = sum(
            1 for sk in skills
            if sk.get("proficiency") == "expert" and sk.get("duration_months", 1) == 0
        )
        if expert_zero >= 3:
            return True

        # Check 3: total career months vs years_of_experience wildly off
        total_months = sum(r.get("duration_months", 0) for r in career)
        declared_months = yoe * 12
        if total_months > declared_months * 1.5 + 24:  # more than 50% over + 2yr buffer
            return True

        # Check 4: skills count > 25 all at expert/advanced with max endorsements
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


# ---------------------------------------------------------------------------
# SKILL SCORING
# ---------------------------------------------------------------------------

def skill_score(c: dict) -> float:
    """0-1: weighted overlap with JD must-have and nice-to-have skills."""
    candidate_skills = set()
    skill_proficiency_bonus = 0.0
    assessment_scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})

    for sk in c.get("skills", []):
        name = sk["name"].lower().strip()
        candidate_skills.add(name)
        # also add partial tokens for compound skill names
        for token in name.split():
            candidate_skills.add(token)

    # also pull from career descriptions
    all_text = " ".join([
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
        *[r.get("description", "") for r in c.get("career_history", [])],
    ]).lower()

    must_hits = 0
    must_total = len(MUST_HAVE_SKILLS)
    for sk in MUST_HAVE_SKILLS:
        if sk in candidate_skills or sk in all_text:
            must_hits += 1

    nice_hits = 0
    nice_total = len(NICE_TO_HAVE_SKILLS)
    for sk in NICE_TO_HAVE_SKILLS:
        if sk in candidate_skills or sk in all_text:
            nice_hits += 1

    must_ratio = must_hits / must_total
    nice_ratio = nice_hits / nice_total

    # Bonus for verified assessment scores on key skills
    assessment_bonus = 0.0
    for key, val in assessment_scores.items():
        k = key.lower()
        if any(s in k for s in ["python", "nlp", "ml", "embedding", "retrieval"]):
            assessment_bonus += (val / 100) * 0.05

    raw = 0.70 * must_ratio + 0.25 * nice_ratio + min(0.1, assessment_bonus)
    return clamp(raw)


# ---------------------------------------------------------------------------
# CAREER FIT SCORING
# ---------------------------------------------------------------------------

def career_score(c: dict) -> float:
    """0-1: title trajectory, company type, experience band, location."""
    profile = c["profile"]
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})
    score = 0.0

    # --- Experience band ---
    yoe = profile.get("years_of_experience", 0)
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        exp_score = 1.0
    elif EXP_MIN <= yoe < EXP_IDEAL_MIN:
        exp_score = 0.7
    elif EXP_IDEAL_MAX < yoe <= EXP_MAX:
        exp_score = 0.8
    elif yoe > EXP_MAX:
        exp_score = 0.5  # over-experienced for founding team
    else:
        exp_score = 0.2
    score += 0.20 * exp_score

    # --- Title relevance ---
    title = profile.get("current_title", "").lower()
    ai_titles = {
        "ml engineer", "machine learning engineer", "ai engineer",
        "nlp engineer", "research engineer", "applied scientist",
        "data scientist", "applied ml", "senior engineer",
        "software engineer", "backend engineer", "full stack",
    }
    title_score = 0.0
    for t in ai_titles:
        if t in title:
            title_score = 1.0
            break
    if not title_score:
        # partial match
        for word in ["engineer", "scientist", "ml", "ai", "nlp", "data"]:
            if word in title:
                title_score = 0.5
                break
    score += 0.20 * title_score

    # --- Product company experience (anti-consulting) ---
    all_text = " ".join([
        c["profile"].get("summary", ""),
        *[r.get("description", "") for r in career],
    ]).lower()

    consulting_penalty = 0.0
    all_consulting = True
    for role in career:
        company = role.get("company", "").lower()
        if any(f in company for f in CONSULTING_FIRMS):
            consulting_penalty += role.get("duration_months", 0)
        else:
            all_consulting = False

    total_career_months = sum(r.get("duration_months", 0) for r in career) or 1
    consulting_fraction = consulting_penalty / total_career_months
    company_score = 1.0 - consulting_fraction * 0.8  # partial penalty
    if all_consulting and len(career) >= 2:
        company_score = 0.1  # hard penalty for full consulting career
    score += 0.15 * company_score

    # --- Production deployment signals in career descriptions ---
    production_keywords = [
        "production", "deployed", "shipped", "at scale", "real users",
        "million", "latency", "inference", "serving", "api", "endpoint",
    ]
    prod_hits = sum(1 for kw in production_keywords if kw in all_text)
    prod_score = clamp(prod_hits / 6)
    score += 0.15 * prod_score

    # --- Location ---
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "")

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

    # --- Notice period ---
    notice = signals.get("notice_period_days", 90)
    if notice <= NOTICE_IDEAL:
        notice_score = 1.0
    elif notice <= NOTICE_OK:
        notice_score = normalize(notice, NOTICE_OK, NOTICE_IDEAL)
    else:
        notice_score = 0.2
    score += 0.10 * notice_score

    # --- Salary fit ---
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 999)
    sal_overlap = (
        min(sal_max, SALARY_JD_MAX) - max(sal_min, SALARY_JD_MIN)
    ) / (SALARY_JD_MAX - SALARY_JD_MIN)
    salary_score = clamp(sal_overlap)
    score += 0.05 * salary_score

    return clamp(score)


# ---------------------------------------------------------------------------
# BEHAVIORAL SIGNAL SCORING
# ---------------------------------------------------------------------------

def behavior_score(c: dict) -> float:
    """0-1: engagement, availability, verification signals."""
    s = c.get("redrob_signals", {})
    score = 0.0

    # Recency / activity
    last_active = parse_date(s.get("last_active_date"))
    days_inactive = days_since(last_active)
    if days_inactive <= 14:
        activity = 1.0
    elif days_inactive <= 30:
        activity = 0.85
    elif days_inactive <= 90:
        activity = 0.6
    elif days_inactive <= 180:
        activity = 0.3
    else:
        activity = 0.05
    score += 0.20 * activity

    # Open to work
    if s.get("open_to_work_flag"):
        score += 0.10

    # Recruiter response rate
    rr = s.get("recruiter_response_rate", 0.0)
    score += 0.15 * rr

    # Response time (lower = better; cap at 48h ideal)
    rt = s.get("avg_response_time_hours", 48)
    rt_score = clamp(1.0 - rt / 48)
    score += 0.05 * rt_score

    # Profile completeness
    pc = s.get("profile_completeness_score", 0) / 100
    score += 0.10 * pc

    # Interview completion
    ic = s.get("interview_completion_rate", 0.0)
    score += 0.10 * ic

    # GitHub activity (good signal for engineering roles)
    gh = s.get("github_activity_score", -1)
    gh_score = (gh / 100) if gh >= 0 else 0.0
    score += 0.10 * gh_score

    # Saved by recruiters — market signal
    saved = min(s.get("saved_by_recruiters_30d", 0), 20)
    score += 0.05 * (saved / 20)

    # Verification
    verified = (
        int(s.get("verified_email", False)) +
        int(s.get("verified_phone", False)) +
        int(s.get("linkedin_connected", False))
    ) / 3
    score += 0.10 * verified

    # Offer acceptance rate (positive signal; -1 means no history)
    oar = s.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += 0.05 * oar

    return clamp(score)


# ---------------------------------------------------------------------------
# SEMANTIC SCORE  (offline TF-IDF cosine similarity — no network required)
# ---------------------------------------------------------------------------

def build_candidate_text(c: dict) -> str:
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


def compute_semantic_scores(candidates: list, batch_size=512) -> np.ndarray:
    """TF-IDF cosine similarity between JD and candidate texts (CPU, offline)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [build_candidate_text(c) for c in candidates]
    corpus = [JD_TEXT] + texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[0]
    cand_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vec, cand_vecs).flatten()
    return sims


# ---------------------------------------------------------------------------
# REASONING GENERATION
# ---------------------------------------------------------------------------

def generate_reasoning(c: dict, rank: int, final_score: float,
                        sem: float, skill: float, career: float, behav: float) -> str:
    p = c["profile"]
    s = c["redrob_signals"]
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "N/A")
    loc = p.get("location", "N/A")
    notice = s.get("notice_period_days", "?")
    response_rate = s.get("recruiter_response_rate", 0)
    last_active = s.get("last_active_date", "unknown")
    gh = s.get("github_activity_score", -1)

    # pick 2-3 most relevant skills
    jd_relevant = []
    for sk in c.get("skills", []):
        n = sk["name"].lower()
        if any(m in n for m in ["embedding", "vector", "nlp", "retrieval", "python",
                                  "transformers", "llm", "rag", "ranking", "faiss",
                                  "search", "bert", "fine-tun", "pytorch"]):
            jd_relevant.append(sk["name"])
    rel_skills_str = ", ".join(jd_relevant[:3]) if jd_relevant else "limited relevant skills"

    # concerns
    concerns = []
    days_inactive = days_since(parse_date(last_active))
    if days_inactive > 90:
        concerns.append(f"inactive {days_inactive}d")
    if response_rate < 0.3:
        concerns.append(f"low response rate ({response_rate:.0%})")
    if notice > 60:
        concerns.append(f"long notice ({notice}d)")
    if s.get("willing_to_relocate") is False and p.get("country", "").lower() != "india":
        concerns.append("not willing to relocate")

    concern_str = f" Concern: {'; '.join(concerns)}." if concerns else ""

    # high rank vs low rank tone
    if rank <= 10:
        tone = f"Strong fit: {title}, {yoe}y exp, skills include {rel_skills_str}."
    elif rank <= 30:
        tone = f"Good fit: {title} with {yoe}y exp; relevant skills: {rel_skills_str}."
    elif rank <= 60:
        tone = f"Moderate fit: {title}, {yoe}y exp; partial skill overlap ({rel_skills_str})."
    else:
        tone = f"Weak fit: {title} with {yoe}y; limited alignment with JD requirements."

    gh_note = f" GitHub score {gh:.0f}/100." if gh > 0 else ""
    return f"{tone}{gh_note} {loc}, notice {notice}d, response rate {response_rate:.0%}.{concern_str}"


# ---------------------------------------------------------------------------
# MAIN RANKER
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list:
    p = Path(path)
    candidates = []
    if p.suffix == ".gz":
        opener = lambda: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda: open(p, "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return candidates


def rank_candidates(candidates: list, top_n: int = 100) -> list:
    print(f"Loaded {len(candidates)} candidates.", file=sys.stderr)

    # Step 1: Honeypot filter (flag but don't remove — still include at bottom)
    honeypot_flags = [is_honeypot(c) for c in candidates]
    n_honeypots = sum(honeypot_flags)
    print(f"Flagged {n_honeypots} honeypot candidates.", file=sys.stderr)

    # Step 2: Fast pre-filter to reduce semantic embedding load
    # Compute skill + career scores for all candidates first (fast)
    print("Computing skill scores...", file=sys.stderr)
    skill_scores = np.array([skill_score(c) for c in candidates])

    print("Computing career scores...", file=sys.stderr)
    career_scores = np.array([career_score(c) for c in candidates])

    print("Computing behavior scores...", file=sys.stderr)
    behavior_scores = np.array([behavior_score(c) for c in candidates])

    # Pre-filter: keep top 5000 by (skill + career) for semantic embedding
    fast_score = 0.50 * skill_scores + 0.50 * career_scores
    top_5k_idx = np.argsort(fast_score)[::-1][:5000]

    print(f"Running semantic embeddings on top {len(top_5k_idx)} candidates...", file=sys.stderr)
    top_5k_candidates = [candidates[i] for i in top_5k_idx]
    sem_scores_5k = compute_semantic_scores(top_5k_candidates)

    # Build full semantic score array (non-top-5k get 0)
    sem_scores = np.zeros(len(candidates))
    for i, idx in enumerate(top_5k_idx):
        sem_scores[idx] = sem_scores_5k[i]

    # Step 3: Combine scores
    weights = dict(semantic=0.35, skill=0.30, career=0.20, behavior=0.15)
    combined = (
        weights["semantic"] * sem_scores +
        weights["skill"] * skill_scores +
        weights["career"] * career_scores +
        weights["behavior"] * behavior_scores
    )

    # Apply honeypot penalty (push them to the bottom)
    for i, flag in enumerate(honeypot_flags):
        if flag:
            combined[i] *= 0.05

    # Step 4: Sort and take top_n
    ranked_idx = np.argsort(combined)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        c = candidates[idx]
        fs = float(combined[idx])
        reasoning = generate_reasoning(
            c, rank, fs,
            float(sem_scores[idx]),
            float(skill_scores[idx]),
            float(career_scores[idx]),
            float(behavior_scores[idx]),
        )
        results.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": round(fs, 6),
            "reasoning": reasoning,
        })

    return results


def write_csv(results: list, out_path: str, team_id: str = "team_submission"):
    p = Path(out_path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Written {len(results)} rows to {p}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path (default: submission.csv)")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of candidates to rank (default: 100)")
    args = parser.parse_args()

    print("Loading candidates...", file=sys.stderr)
    candidates = load_candidates(args.candidates)

    results = rank_candidates(candidates, top_n=args.top)
    write_csv(results, args.out)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
