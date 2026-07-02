# Redrob Hackathon — Narrative Drift-Aware Candidate Ranker

## Approach

This system ranks candidates not on keyword overlap, but on a multi-signal composite score that mirrors how a seasoned recruiter actually reads a profile. The core thesis: **a candidate is a trajectory, not a static document.** We score them on where they're heading, whether they're actually available, and whether their career pattern fits the role's culture — not just whether the right words appear in their skills section.

### Why TF-IDF over embeddings?

The compute constraint (5 min, CPU, no network) rules out cloud LLMs and makes large transformer models marginal. TF-IDF with bigrams + sublinear TF weighting captures the key vocabulary of the JD (retrieval, embeddings, vector search, ranking, production) and scores candidates on contextual text overlap accurately within milliseconds. The remaining budget is spent on the structured signals that actually differentiate strong candidates.

---

## Scoring Architecture

Final score = **0.35 × semantic** + **0.30 × skill** + **0.20 × career** + **0.15 × behavior**

### 1. Semantic Score (35%)
TF-IDF cosine similarity between the JD text and a candidate's concatenated narrative (headline + summary + role descriptions + skills). Uses bigrams and sublinear TF to penalise keyword stuffing. Computed over the full 100K pool.

### 2. Skill Score (30%)
Weighted overlap against two skill lists derived from the JD:

- **Must-have skills** (70% weight): sentence-transformers, vector DBs (FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch), NLP, LLMs, RAG, fine-tuning, Python, NDCG/MRR/MAP evaluation frameworks
- **Nice-to-have skills** (25% weight): PyTorch, XGBoost, HuggingFace, FastAPI, Docker, recommendation systems, open source
- **Assessment bonus** (5%): Redrob platform skill assessment scores on relevant skills

Matching is done against skill name tokens AND free text in career descriptions — catching candidates who have the capability but haven't listed the exact buzzword.

### 3. Career Score (20%)
Sub-components:
- **Experience band** (20%): 6–8 years ideal, 5–9 acceptable, outside penalised
- **Title relevance** (20%): ML/AI/NLP/software engineering titles score high
- **Anti-consulting filter** (15%): fraction of career at TCS/Infosys/Wipro/Accenture/Capgemini penalised proportionally; full consulting career → 0.1
- **Production signals** (15%): "deployed", "at scale", "real users", "latency", "serving" in role descriptions
- **Location** (15%): Pune/Noida/Delhi/Hyderabad/Mumbai/Bangalore full score; rest of India + willing to relocate partial; outside India further penalised
- **Notice period** (10%): ≤30 days ideal, ≤90 acceptable, 90+ penalised
- **Salary fit** (5%): overlap with estimated JD range (₹25–80 LPA)

### 4. Behavioral Score (15%)
Redrob platform signals as an availability/engagement multiplier:
- **Recency** (20%): days since last login; inactive >90 days heavily penalised
- **Open to work** (10%): binary flag
- **Recruiter response rate** (15%): most predictive availability signal
- **Response time** (5%): lower is better
- **Profile completeness** (10%): proxy for seriousness
- **Interview completion rate** (10%): reliability signal
- **GitHub activity** (10%): engineering credibility for this role
- **Saved by recruiters** (5%): market validation
- **Verification** (10%): email + phone + LinkedIn connected
- **Offer acceptance rate** (5%): historical reliability

### 5. Honeypot Detection
Before scoring, profiles are flagged as honeypots if they have:
- Career duration at a company longer than the company's age (impossible tenure)
- 3+ skills claimed as "expert" with 0 months of usage
- Total career months >150% of declared YOE
- 15+ skills at advanced/expert with suspiciously uniform max endorsements

Flagged candidates receive a 0.05× score multiplier, pushing them below the top 100 cutoff without fully removing them from the pool.

---

## Running the Ranker

### Setup
```bash
pip install -r requirements.txt
```

### Full ranking (produces submission.csv)
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```
Also accepts uncompressed `.jsonl`:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Runtime
- ~2–3 minutes on 16GB CPU for 100K candidates
- No GPU required, no network calls

---

## Repository Structure

```
.
├── rank.py                        # Main ranker — single file, no hidden steps
├── requirements.txt
├── README.md
├── submission_metadata.yaml       # Portal metadata
└── sample_candidates.jsonl        # 50-candidate test sample
```

---

## Design Decisions & Trade-offs

**Why not call an LLM per candidate?** The compute constraint (5 min, CPU, no network) makes this impossible for 100K candidates. More importantly, LLM calls per candidate don't scale to production — a system that works at 100K should work at 1M.

**Why TF-IDF over sentence-transformers?** Sentence-transformers need network access to download model weights (or a pre-downloaded cache that must be shipped with the repo). TF-IDF is deterministic, reproducible, and adds zero dependencies beyond scikit-learn. The quality trade-off is real but acceptable given the structured scoring covers most of the semantic gap.

**Why pre-filter to top 5K for semantic scoring?** TF-IDF is fast enough that this pre-filter isn't strictly needed, but it mirrors the architecture that would be needed with neural embeddings — fast first-stage retrieval, then slower re-ranking. It's the production-ready pattern.

**Why 35/30/20/15 weights?** Semantic captures the overall narrative fit. Skill is the most direct signal but over-weights keyword stuffers, so it's capped at 30%. Career trajectory is the anti-gaming layer — a Marketing Manager can have all the AI keywords but fails the title/company filter. Behavioral is the availability multiplier — a perfect-on-paper candidate who hasn't logged in for 6 months is not actually hireable.

---

## AI Tools Declaration

This project was built with the help of Claude (Anthropic) for architecture discussion and code review. All engineering decisions, scoring weights, and JD analysis were made by the myself. No candidate data was fed to any external LLM during ranking.