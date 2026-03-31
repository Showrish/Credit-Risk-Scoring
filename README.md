# CreditIQ — Mortgage Risk Scoring

An end-to-end machine learning system that predicts mortgage loan approval outcomes using 8.6 million real loan applications from the FFIEC/CFPB Home Mortgage Disclosure Act (HMDA) 2024 national dataset. The model classifies whether a mortgage application will be approved or denied, with full SHAP-based explainability and bias auditing across demographic groups.

![Dashboard](screenshots/dashboard.png)

---

## Why This Matters

Mortgage approval models are deployed in production at every major bank and lender. The Fair Housing Act and Equal Credit Opportunity Act legally require lenders to explain denials and audit for demographic bias. This project mirrors that real-world requirement — building not just a predictive model, but an explainable and auditable one.

---

## Tech Stack

- **Data:** FFIEC/CFPB HMDA 2024 National Snapshot (12.2M raw records)
- **Storage & Querying:** DuckDB
- **Modeling:** LightGBM, XGBoost, Random Forest, Logistic Regression
- **Explainability:** SHAP TreeExplainer
- **Backend:** FastAPI
- **Frontend:** HTML/CSS/JS (Netflix-inspired dark UI)
- **Dashboard:** Power BI
- **Environment:** Python, Google Colab, pandas, scikit-learn

---

## Project Structure

```
Credit Scoring/
├── api/
│   ├── app.py
│   ├── models/
│   │   ├── lgbm_model.pkl
│   │   ├── le_dict.pkl
│   │   └── feature_cols.pkl
│   └── static/
│       └── index.html
├── dashboard/
│   └── credit_risk_dashboard.pbix
├── data/
│   ├── hmda.duckdb
│   ├── hmda_processed.csv
│   └── queries/
│       ├── approval_by_income.sql
│       ├── approval_by_state.sql
│       ├── denial_by_race.sql
│       ├── dti_vs_denial.sql
│       └── loan_amount_vs_approval.sql
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── screenshots/
└── README.md
```

---

## Data Pipeline

- Downloaded 2024 HMDA national snapshot from [ffiec.cfpb.gov](https://ffiec.cfpb.gov)
- Raw file: 12.2M rows, 99 columns — all stored as VARCHAR due to mixed values like "Exempt"
- Loaded into DuckDB using `read_csv_auto` with `all_varchar=1`
- Filtered to approved (action_taken 1, 2) and denied (action_taken 3, 7) loans only
- Final dataset: **8.68M rows**, binary target (1=approved, 0=denied), 75/25 class split
- Replaced "Exempt" and "NA" strings with NaN
- Log-transformed skewed columns: loan_amount, income, property_value
- Parsed debt_to_income_ratio from range strings (e.g. "20%-<30%") into numeric values
- Dropped post-approval leaky columns: interest_rate, rate_spread, loan_term

---

## Model Comparison

All models trained on 6.94M samples (80/20 stratified split). Logistic Regression and Random Forest trained on 500K subsample due to scale.

| Model | ROC-AUC | Accuracy | F1 (Denied) | Train Time |
|---|---|---|---|---|
| **LightGBM** | **0.877** | 0.819 | **0.670** | 189s |
| XGBoost | 0.868 | 0.834 | 0.516 | 130s |
| Random Forest | 0.853 | 0.846 | 0.618 | 77s |
| Logistic Regression | 0.750 | 0.706 | 0.528 | 72s |

**LightGBM selected as final model** — best ROC-AUC and best F1 on the denied class (the harder, more important class for real-world use).

---

## SHAP Explainability

SHAP TreeExplainer run on 10,000 test samples to explain individual and global predictions.

![SHAP Beeswarm](screenshots/shap_beeswarm.png)
![SHAP Importance](screenshots/shap_importance.png)

**Key findings:**
- **DTI is the #1 driver** — high debt-to-income ratio strongly pushes predictions towards denial
- **Loan purpose and property value** are the next most influential financial features
- **Race has measurable SHAP impact** — indicating potential systemic bias present in historical lending data
- **Sex and ethnicity** show smaller but non-zero effects — relevant for regulatory fairness audits
- **Geographic features** (county, state) have meaningful impact, suggesting regional lending disparities

---

## SQL Analysis

Five analytical queries run directly on DuckDB against 8.68M records:

**Approval Rate by Income Segment**
| Income | Approval Rate |
|---|---|
| Over $200K | 84.3% |
| $100K–$200K | 80.8% |
| $50K–$100K | 72.8% |
| Under $50K | 50.9% |

**Denial Rate by Race**
| Race | Denial Rate |
|---|---|
| Black or African American | 37.3% |
| American Indian or Alaska Native | 37.6% |
| Native Hawaiian or Pacific Islander | 39.1% |
| White | 22.2% |
| Asian | 21.7% |

**DTI vs Denial Rate**
| DTI Range | Denial Rate |
|---|---|
| >60% | 91.7% |
| 50%–60% | 39.4% |
| 30%–36% | 14.5% |
| <20% | 29.2% |

**Approval Rate by Loan Size**
| Loan Size | Approval Rate |
|---|---|
| $500K–$1M | 86.6% |
| $300K–$500K | 84.9% |
| Under $150K | 64.1% |

---

## Business Insights

- **DTI above 60% is effectively a hard denial** — 91.7% denial rate. Underwriting thresholds should be clearly communicated to applicants before application.
- **Low-income applicants are disproportionately denied** — 33 percentage point gap between under $50K and over $200K income segments.
- **Racial lending disparity is statistically significant** — Black and Native American applicants denied at nearly 15 percentage points higher than White applicants, confirmed by both SQL analysis and SHAP values.
- **Larger loans get approved more easily** — counterintuitive but reflects that high-value properties in strong markets carry less risk for lenders.
- **Regional disparities exist** — Midwest states (ND, IA, MN) approve at 83–85% vs Southern states (FL, LA, MS) at 69–71%.

---

## API

FastAPI backend with single and batch prediction endpoints, each returning a decision, probability, and top SHAP factors.

**Run locally:**
```bash
cd api
uvicorn app:app --reload
```

Open `http://localhost:8000` for the UI or `http://localhost:8000/docs` for Swagger.

**Single prediction — POST /predict**
```json
{
  "loan_amount": 320000,
  "income": 85,
  "property_value": 420000,
  "dti": 35,
  "loan_type": "1",
  "loan_purpose": "1",
  "lien_status": "1",
  "derived_race": "White",
  "derived_ethnicity": "Not Hispanic or Latino",
  "derived_sex": "Male",
  "applicant_age": "35-44",
  "applicant_age_above_62": "No",
  "state_code": "MA",
  "county_code": "25017",
  "conforming_loan_limit": "C",
  "derived_loan_product_type": "Conventional:First Lien",
  "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
  "occupancy_type": "1",
  "preapproval": "2"
}
```

**Response:**
```json
{
  "decision": "Approved",
  "probability": 0.8741,
  "top_factors": {
    "dti": 0.312,
    "loan_purpose": 0.187,
    "property_value_log": 0.143,
    "loan_amount_log": 0.098,
    "derived_race": -0.071
  }
}
```

**Batch prediction — POST /predict/batch**

Accepts a list of applications and returns predictions for all.

![Single Prediction](screenshots/single_prediction.png)
![Batch Prediction](screenshots/batch_prediction.png)

---

## Power BI Dashboard

Interactive dashboard built on the full 8.68M record dataset with slicers for state filtering.

![Dashboard](screenshots/dashboard.png)

**Visuals:**
- KPI cards: Total Applications, Approval Rate, Denial Rate, Avg Loan Amount
- Denial Rate by Race — horizontal bar
- Mortgage Approval Rate by State — filled map
- Approval Rate by Income Segment — column chart
- Denial Rate by DTI Bracket — column chart
- Approval Rate by Loan Size — column chart

---

## Setup

```bash
pip install fastapi uvicorn lightgbm shap pandas numpy scikit-learn joblib duckdb
```

Place model files in `api/models/` and run:

```bash
cd api
uvicorn app:app --reload
```

---

## Data Source

FFIEC/CFPB Home Mortgage Disclosure Act (HMDA) 2024 National Snapshot  
[https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024)