from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import shap

app = FastAPI(title="Credit Risk Scoring API")

# Load model and encoders
model = joblib.load("models/lgbm_model.pkl")
le_dict = joblib.load("models/le_dict.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
explainer = shap.TreeExplainer(model)

cat_cols = list(le_dict.keys())

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


class LoanApplication(BaseModel):
    loan_amount: float
    income: float
    property_value: float
    dti: Optional[float] = None
    loan_type: str = "1"
    loan_purpose: str = "1"
    lien_status: str = "1"
    derived_race: str = "White"
    derived_ethnicity: str = "Not Hispanic or Latino"
    derived_sex: str = "Male"
    applicant_age: str = "35-44"
    applicant_age_above_62: str = "No"
    state_code: str = "MA"
    county_code: str = "25017"
    conforming_loan_limit: str = "C"
    derived_loan_product_type: str = "Conventional:First Lien"
    derived_dwelling_category: str = "Single Family (1-4 Units):Site-Built"
    occupancy_type: str = "1"
    preapproval: str = "2"


class BatchRequest(BaseModel):
    applications: List[LoanApplication]


def preprocess(app: LoanApplication):
    d = app.dict()
    d['loan_amount_log'] = np.log1p(d['loan_amount'])
    d['income_log'] = np.log1p(d['income'])
    d['property_value_log'] = np.log1p(d['property_value'])
    del d['loan_amount'], d['income'], d['property_value']

    for col in cat_cols:
        le = le_dict[col]
        val = str(d[col])
        if val in le.classes_:
            d[col] = int(le.transform([val])[0])
        else:
            d[col] = 0

    df = pd.DataFrame([d])[feature_cols]
    return df


@app.post("/predict")
def predict(application: LoanApplication):
    df = preprocess(application)
    prob = float(model.predict_proba(df)[0][1])
    decision = "Approved" if prob >= 0.5 else "Denied"
    shap_vals = explainer.shap_values(df)[0]
    top_factors = dict(
        sorted(zip(feature_cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
    )
    return {
        "decision": decision,
        "probability": round(prob, 4),
        "top_factors": top_factors
    }


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for i, application in enumerate(request.applications):
        df = preprocess(application)
        prob = float(model.predict_proba(df)[0][1])
        decision = "Approved" if prob >= 0.5 else "Denied"
        shap_vals = explainer.shap_values(df)[0]
        top_factors = dict(
            sorted(zip(feature_cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        results.append({
            "application_id": i + 1,
            "decision": decision,
            "probability": round(prob, 4),
            "top_factors": top_factors
        })
    return {"results": results}