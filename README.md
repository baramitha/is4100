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

## Repo Structure
IS4100-ML-Pipeline/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/          # Original Kaggle dataset (not uploaded)
│   └── processed/    # Cleaned dataset output
│
├── src/
│   ├── data_engineering.py    # SCRUM-14, 15, 16
│   ├── feature_engineering.py # SCRUM-17
│   ├── model.py               # SCRUM-7
│   └── train_test_split.py    # SCRUM-18
│
└── notebooks/
└── exploration.ipynb      # SCRUM-14: Data exploration
