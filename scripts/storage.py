from pathlib import Path

class Storage:
    RANDOM_SEED = 42

    TRAIN_SKIP_ROW_COUNT = 31_500

    DATASET_PATH = Path(".\dataset")
    ORIGINAL_DATASET_FILE_PATH = DATASET_PATH / "student-mat.csv"
    DIVIDED_DATASET_PATH = DATASET_PATH / "divided"
    FEATURES_DATASET_PATH = DATASET_PATH / "features"
    MODELS_PATH = DATASET_PATH / "models"

    RESULT_FILE_PATH = Path("result.json")
