import sys
import os
import logging
import yaml

# Fix paths properly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

sys.path.append(BASE_DIR)

from data_prep import load_data, clean_data, encode_data, split_data
from train import train

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load config safely
config_path = os.path.join(ROOT_DIR, "config.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)


# ─────────────────────────────
# DATA QUALITY CHECK
# ─────────────────────────────
def check_data_quality(df):
    logger.info("Running data quality checks...")
    issues = []

    if df.shape[0] < 1000:
        issues.append(f"Not enough rows: {df.shape[0]}")

    if df.isnull().sum().sum() > 0:
        issues.append("Missing values found")

    target = config["data"]["target_column"]
    if target not in df.columns:
        issues.append(f"Target column '{target}' not found")

    churn_rate = df[target].mean()
    if churn_rate < 0.05 or churn_rate > 0.60:
        issues.append(f"Unusual churn rate: {churn_rate:.2%}")

    if issues:
        for issue in issues:
            logger.error(issue)
        return False

    logger.info("✅ Data quality passed")
    return True


# ─────────────────────────────
# MODEL QUALITY CHECK
# ─────────────────────────────
def check_model_quality(metrics: dict):
    logger.info("Checking model quality...")

    if metrics is None:
        raise ValueError("Metrics is None — training failed")

    issues = []

    if metrics["accuracy"] < 0.75:
        issues.append(f"Accuracy too low: {metrics['accuracy']:.3f}")

    if metrics["f1_score"] < 0.60:
        issues.append(f"F1 too low: {metrics['f1_score']:.3f}")

    if metrics["auc_roc"] < 0.75:
        issues.append(f"AUC too low: {metrics['auc_roc']:.3f}")

    if issues:
        for issue in issues:
            logger.error(issue)
        return False

    logger.info("✅ Model quality passed")
    return True


# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────
def run_pipeline():
    logger.info("=" * 50)
    logger.info("🚀 Pipeline Started")
    logger.info("=" * 50)

    # Stage 1 — Load data
    try:
        df = load_data()
        logger.info(f"Data loaded: {df.shape}")
    except Exception as e:
        logger.error(e)
        return False

    # Stage 2 — Data quality
    if not check_data_quality(df):
        logger.error("Pipeline stopped — bad data")
        return False

    # Stage 3 — Prep
    try:
        df = clean_data(df)
        df = encode_data(df)
        X, y = split_data(df)
        logger.info(f"Prepared data: {X.shape}")
    except Exception as e:
        logger.error(e)
        return False

    # Stage 4 — Train
    try:
        metrics = train()
        logger.info(f"Training done: {metrics}")
    except Exception as e:
        logger.error(e)
        return False

    # Stage 5 — Validate
    if not check_model_quality(metrics):
        logger.error("Pipeline stopped — bad model")
        return False

    logger.info("✅ Pipeline SUCCESS")
    return True


if __name__ == "__main__":
    success = run_pipeline()

    if not success:
        sys.exit(1)