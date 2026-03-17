from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Bank Churn Model API running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e)}