import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

def load_data():
    df = pd.read_csv(config["data"]["filepath"])
    print("✅ Data loaded:", df.shape)
    return df

def clean_data(df):
    # Drop useless columns
    df = df.drop(columns=config["data"]["drop_columns"])
    print("✅ Dropped useless columns")
    return df

def encode_data(df):
    # Convert text columns to numbers
    # Machine learning only understands numbers!

    le = LabelEncoder()

    df["Geography"] = le.fit_transform(df["Geography"])
    # France=0, Germany=1, Spain=2

    df["Gender"] = le.fit_transform(df["Gender"])
    # Female=0, Male=1

    df["Card Type"] = le.fit_transform(df["Card Type"])
    # DIAMOND=0, GOLD=1, PLATINUM=2, SILVER=3

    print("✅ Text columns converted to numbers")
    return df

def split_data(df):
    # Separate features (X) from target (y)
    target = config["data"]["target_column"]

    X = df.drop(columns=[target])  # everything except Exited
    y = df[target]                 # only Exited column

    print("✅ Features shape:", X.shape)
    print("✅ Target shape:", y.shape)
    return X, y

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)
    X, y = split_data(df)
    print("\n🎯 Data is ready for training!")
    print("Features:", X.columns.tolist())