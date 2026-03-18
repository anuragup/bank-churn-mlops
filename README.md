# Churn Prediction – MLOps Basics

## Overview
Simple churn prediction model with basic MLOps setup for retraining and artifact handling.

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