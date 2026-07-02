# Narrative Drift-Aware Candidate Ranker

## Technical Documentation

## 1. Problem Statement

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword overlap between job descriptions and candidate profiles. This approach often favors candidates who optimize resumes for search engines rather than those who represent the strongest overall fit for a role.

The objective of this project is to develop a scalable candidate ranking system capable of evaluating over 100,000 candidates under strict compute constraints while incorporating richer signals such as career trajectory, behavioral engagement, and profile authenticity.

---

## 2. Proposed Solution

We propose a Narrative Drift-Aware Candidate Ranker that evaluates candidates as evolving career trajectories rather than static resumes.

The system combines:

* Semantic relevance
* Technical skill alignment
* Career progression analysis
* Behavioral engagement signals
* Profile authenticity validation

These signals are aggregated into a composite score that approximates recruiter decision-making more closely than traditional keyword matching systems.

---

## 3. Cross-Domain Adaptation

The ranking framework was originally developed for financial narrative intelligence applications where it tracked changes in company narratives and market sentiment over time.

For this challenge, the same intelligence engine was adapted to model candidate narratives and career evolution.

| Financial Intelligence | Talent Intelligence |
| ---------------------- | ------------------- |
| Company Narrative      | Candidate Narrative |
| Narrative Drift        | Career Trajectory   |
| Market Signals         | Hiring Signals      |
| Fraud Detection        | Honeypot Detection  |
| Asset Ranking          | Candidate Ranking   |

---

## 4. System Architecture

The ranking pipeline consists of the following stages:

1. Job Description Ingestion
2. Requirement Extraction
3. Candidate Retrieval
4. Multi-Signal Scoring
5. Validation and Explainability
6. Composite Ranking
7. Ranked Candidate Output

---

## 5. Candidate Retrieval

Candidate retrieval is performed using TF-IDF vectorization with:

* Unigrams and bigrams
* Sublinear term frequency scaling
* Cosine similarity scoring

The candidate narrative consists of:

* Headline
* Summary
* Experience descriptions
* Skills
* Career history

This provides contextual matching while remaining computationally efficient under CPU-only constraints.

---

## 6. Multi-Signal Ranking

Final score:

Final Score = 0.35 × Semantic + 0.30 × Skill + 0.20 × Career + 0.15 × Behavioral

### Semantic Score (35%)

Measures contextual similarity between candidate narratives and the job description using TF-IDF cosine similarity.

### Skill Score (30%)

Evaluates overlap between candidate capabilities and required skills extracted from the JD.

### Career Score (20%)

Considers:

* Experience band
* Job titles
* Production deployment experience
* Geographic fit
* Notice period
* Compensation alignment

### Behavioral Score (15%)

Uses platform engagement signals such as:

* Recency of activity
* Recruiter response rate
* Profile completeness
* Interview completion rate
* Open-to-work status

---

## 7. Data Validation and Explainability

The system uses deterministic scoring rules to ensure every ranking decision can be explained.

Each candidate receives:

* Semantic score
* Skill score
* Career score
* Behavioral score
* Final composite score

No generative explanations or unsupported justifications are used.

---

## 8. Honeypot Detection

Potentially suspicious profiles are identified using rule-based validation:

* Impossible employment durations
* Inflated experience claims
* Skill inconsistencies
* Unrealistic endorsement distributions

Flagged candidates receive a ranking penalty multiplier rather than complete removal.

---

## 9. Runtime Performance

Performance targets:

* Candidate pool: 100,000+
* Runtime: ~2–3 minutes
* Hardware: CPU only
* Network calls: 0
* GPU requirement: None

The system satisfies all challenge runtime and infrastructure constraints.

---

## 10. Technology Stack

### Language

* Python

### Libraries

* scikit-learn
* pandas
* numpy
* gzip

### Algorithms

* TF-IDF
* Cosine Similarity
* Weighted Score Aggregation
* Rule-Based Validation

---

## 11. Future Improvements

Potential future enhancements include:

* Lightweight local embedding models
* Learning-to-rank algorithms
* Dynamic score calibration
* Recruiter feedback loops
* Domain-specific fine-tuning

---

## 12. Conclusion

The Narrative Drift-Aware Candidate Ranker demonstrates that effective hiring decisions require more than keyword matching. By combining semantic understanding with career, behavioral, and authenticity signals, the system produces scalable and explainable candidate rankings while remaining compatible with strict production constraints.
