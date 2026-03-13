import pandas as pd

df = pd.read_csv("data/bank_churn.csv")

print("✅ Shape:", df.shape)
print("\n📋 Columns:", df.columns.tolist())
print("\n🔍 First 3 rows:")
print(df.head(3))
print("\n❓ Missing values:")
print(df.isnull().sum())
print("\n📊 Churn distribution:")
print(df["Exited"].value_counts())