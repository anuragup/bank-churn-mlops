def train():
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # ... all your training code ...

    with mlflow.start_run():
        # ... model training ...

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)

        mlflow.sklearn.log_model(model, "churn_model")

        print(f"Accuracy : {accuracy:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(f"AUC ROC  : {auc:.3f}")

        # ✅ return is INSIDE train() function
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc_roc": auc
        }


# this is OUTSIDE — completely separate
if __name__ == "__main__":
    train()