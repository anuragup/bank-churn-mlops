from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

from data_prep import load_data
import pipeline


def train():
    # Load data
    data = load_data()

    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    # Split data (IMPORTANT)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"✅ Model Accuracy: {accuracy}")
    print(f"✅ Model F1 Score: {f1}")
    print(f"✅ Model AUC ROC: {auc}")

    # Save model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "../artifacts/model.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(pipeline, model_path)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc
    }



if __name__ == "__main__":
    train()