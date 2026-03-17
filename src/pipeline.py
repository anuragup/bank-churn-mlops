import sys
import os
import logging
import yaml
import pandas as pd

# Add src folder to path so we can import our files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import load_data, clean_data, encode_data, split_data
from train import train

# Setup logging — professional way to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)


# ─────────────────────────────────────────
# STAGE 1 — Data Quality Checks
# ─────────────────────────────────────────
def check_data_quality(df):
    """
    Check if training data has issues
    Returns True if data is good, False if bad
    Pipeline stops if this returns False
    """
    logger.info("Running data quality checks...")
    issues = []

    # Check 1 — enough rows?
    if df.shape[0] < 1000:
        issues.append(f"❌ Not enough rows: {df.shape[0]} (need 1000+)")

    # Check 2 — missing values?
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"❌ {missing} missing values found")

    # Check 3 — target column exists?
    target = config["data"]["target_column"]
    if target not in df.columns:
        issues.append(f"❌ Target column '{target}' not found")

    # Check 4 — churn rate normal?
    churn_rate = df[target].mean()
    if churn_rate < 0.05 or churn_rate > 0.60:
        issues.append(f"❌ Unusual churn rate: {churn_rate:.2%}")

    # Check 5 — expected columns present?
    expected_cols = [
        "CreditScore", "Age", "Balance",
        "Exited", "Complain", "Satisfaction Score"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"❌ Missing columns: {missing_cols}")

    # Check 6 — no duplicate rows?
    duplicates = df.duplicated().sum()
    if duplicates > 100:
        issues.append(f"❌ Too many duplicate rows: {duplicates}")

    # Results
    if issues:
        logger.error("Data quality checks FAILED:")
        for issue in issues:
            logger.error(issue)
        return False

    logger.info("✅ All data quality checks passed!")
    return True


# ─────────────────────────────────────────
# STAGE 2 — Validate Metrics After Training
# ─────────────────────────────────────────
def check_model_quality(metrics: dict):
    """
    Check if trained model meets minimum performance
    Returns True if model is good enough to deploy
    """
    logger.info("Checking model quality...")
    issues = []

    # Minimum acceptable metrics
    if metrics["accuracy"] < 0.75:
        issues.append(f"❌ Accuracy too low: {metrics['accuracy']:.3f}")

    if metrics["f1_score"] < 0.60:
        issues.append(f"❌ F1 Score too low: {metrics['f1_score']:.3f}")

    if metrics["auc_roc"] < 0.75:
        issues.append(f"❌ AUC ROC too low: {metrics['auc_roc']:.3f}")

    if issues:
        logger.error("Model quality checks FAILED:")
        for issue in issues:
            logger.error(issue)
        return False

    logger.info("✅ Model quality checks passed!")
    return True


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def run_pipeline():
    logger.info("=" * 50)
    logger.info("🚀 Bank Churn MLOps Pipeline Started")
    logger.info("=" * 50)

    # ── Stage 1: Load Data ──────────────────
    logger.info("📂 Stage 1: Loading data...")
    try:
        df = load_data()
        logger.info(f"✅ Data loaded: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False

    # ── Stage 2: Data Quality Check ─────────
    logger.info("🔍 Stage 2: Checking data quality...")
    data_is_good = check_data_quality(df)

    if not data_is_good:
        logger.error("🚨 Pipeline stopped — fix data issues first!")
        return False

    # ── Stage 3: Prepare Data ───────────────
    logger.info("🧹 Stage 3: Preparing data...")
    try:
        df = clean_data(df)
        df = encode_data(df)
        X, y = split_data(df)
        logger.info(f"✅ Features: {X.shape} | Target: {y.shape}")
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return False

    # ── Stage 4: Train Model ─────────────────
    logger.info("🤖 Stage 4: Training model...")
    try:
        metrics = train()

        if metrics["accuracy"] < 0.75:
            raise Exception("Model accuracy too low")
        logger.info(f"✅ Training complete: {metrics}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

    # ── Stage 5: Model Quality Check ─────────
    logger.info("📊 Stage 5: Validating model quality...")
    model_is_good = check_model_quality(metrics)

    if not model_is_good:
        logger.error("🚨 Pipeline stopped — model not good enough!")
        logger.error("Try: more data, tune hyperparameters, different model")
        return False

    # ── Done! ────────────────────────────────
    logger.info("=" * 50)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("🎯 Model is ready for deployment")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    success = run_pipeline()

    # Exit with error code if pipeline failed
    # GitHub Actions uses this to know if pipeline passed or failed
    if not success:
        sys.exit(1)