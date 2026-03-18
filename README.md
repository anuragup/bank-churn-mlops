# Churn Prediction – MLOps 

## Overview
Churn prediction model with MLOps setup for retraining and artifact handling.

---

## Setup

- **Model**: scikit-learn  
- **Artifact**: `model.pkl`  
- **Preprocessing**: saved and reused during inference  
- **CI/CD**: GitHub Actions for retraining  

---

## Flow
Data → Preprocess → Train → Save Model → Predict


---

## Tech

Python · scikit-learn · GitHub Actions 


---

## API (FastAPI)

Basic inference can be exposed via FastAPI:

```python
from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}
