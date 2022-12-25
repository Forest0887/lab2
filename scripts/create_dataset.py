import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import date

from storage import Storage

Storage.DIVIDED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(str(Storage.ORIGINAL_DATASET_FILE_PATH), on_bad_lines='skip', encoding_errors='ignore')

np.random.seed(Storage.RANDOM_SEED)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=Storage.RANDOM_SEED)

df_train.to_csv(str(Storage.DIVIDED_DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Storage.DIVIDED_DATASET_PATH / "test.csv"), index=None)
