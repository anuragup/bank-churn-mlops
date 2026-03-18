Churn Prediction – MLOps Basics
Overview

Simple churn prediction model with basic MLOps setup for retraining and artifact handling.

Setup

Model: scikit-learn

Artifact: model.pkl

Preprocessing: saved and reused during inference

CI/CD: GitHub Actions for retraining

Flow
Data → Preprocess → Train → Save Model → Predict
Notes

Ensures training-serving consistency

Retraining pipeline is CI/CD driven

Ready to integrate with external storage (S3 / Blob)

Tech

Python · scikit-learn · GitHub Actions