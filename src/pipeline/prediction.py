import os
import sys
import warnings
import pandas as pd
import pickle
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

from src.pipeline.training import build_preprocessor
from src.exception import custom_exception
from src.logger import logging

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "artifacts" / "preprocessing.pkl"
DATA_PATH = ROOT_DIR / "notebook" / "data" / "stud.csv"


def load_objects(path: str):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def load_preprocessor(path: str):
    try:
        return load_objects(path)
    except Exception as exc:
        logging.warning(
            "Unable to load preprocessor, rebuilding from data file: %s", exc, exc_info=True
        )
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "Cannot rebuild preprocessor: data file not found at notebook/data/stud.csv"
            )
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=["math_score"])
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)
        return preprocessor


def predict_from_input(
    gender: str,
    race_ethnicity: str,
    parental_level_of_education: str,
    lunch: str,
    test_preparation_course: str,
    reading_score: float,
    writing_score: float,
) -> float:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run /train first.")

    model = load_objects(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)

    input_df = pd.DataFrame(
        [
            {
                "gender": gender,
                "race_ethnicity": race_ethnicity,
                "parental_level_of_education": parental_level_of_education,
                "lunch": lunch,
                "test_preparation_course": test_preparation_course,
                "reading_score": reading_score,
                "writing_score": writing_score,
            }
        ]
    )

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)
    score = float(prediction[0])
    score = max(0.0, min(100.0, score))
    logging.info("Prediction completed successfully: %s", score)
    return score
