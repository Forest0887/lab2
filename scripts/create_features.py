import pandas as pd
import sys

from storage import Storage

Storage.FEATURES_DATASET_PATH.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Storage.DIVIDED_DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Storage.DIVIDED_DATASET_PATH / "test.csv"))

def extract_features(df):
    df["mid_edu"] = (df["Medu"] + df["Fedu"]) / 2 
    return df[["age", "mid_edu"]]

test_features = extract_features(test_df)
train_features = extract_features(train_df)

test_features.to_csv(str(Storage.FEATURES_DATASET_PATH / "test_features.csv"), index=None)
train_features.to_csv(str(Storage.FEATURES_DATASET_PATH / "train_features.csv"), index=None)

test_df.failures.to_csv(str(Storage.FEATURES_DATASET_PATH / "test_label.csv"), index=None)
train_df.failures.to_csv(str(Storage.FEATURES_DATASET_PATH / "train_label.csv"), index=None)
