# IS4100 – ML Pipeline (Group 4)

## Overview
A machine learning pipeline built using Python, managed with 
Agile/Scrum via Jira. The goal is to train a predictive model 
on a Kaggle dataset to support data-driven decision-making.

**Team:** Darryl (Scrum Master) · Naura (Developer)  
**Sprint 1:** Mar 31 – Apr 14 · **Sprint 2:** Apr 14 – Apr 28

---

## Dataset
- **Source:** Kaggle  
- **Task:** Classification (Logistic Regression / Random Forest)  
- Raw data stored in `data/raw/` — not committed to GitHub  

---

## Tech Stack
- **Python** — Pandas, Scikit-Learn
- **Jira** — Sprint & backlog tracking
- **Slack** — Team communication
- **AWS** — Compute resources

---

## Commit Convention
Every commit must reference a Jira ticket:
git commit -m "SCRUM-16: Clean dataset and remove nulls"

---

## How to Run
```bash
pip install pandas scikit-learn
python src/data_engineering.py
python src/model.py
```

---
