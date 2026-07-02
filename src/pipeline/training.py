import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.utils import save_objects
from src.exception import custom_exception
from src.logger import logging

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "artifacts" / "preprocessing.pkl"


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(exclude="object").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def train_model() -> str:
    try:
        df = pd.read_csv(os.path.join("notebook", "data", "stud.csv"))
        target = "math_score"

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        preprocessor = build_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_transformed, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        save_objects(MODEL_PATH, model)
        save_objects(PREPROCESSOR_PATH, preprocessor)

        logging.info(f"Model trained and saved to {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        raise custom_exception(e, sys)
