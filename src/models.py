"""
Model architecture, ensemble training, probability calibration, and odds generation module.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import LABEL_TO_INT, INT_TO_LABEL


CATEGORICAL_COLS = [
    "competition_type",
    "home_team_kind",
    "away_team_kind",
    "home_team_confederation",
    "away_team_confederation",
]

EXCLUDE_COLUMNS = {
    "game_id",
    "home_club_id",
    "away_club_id",
    "home_game_id",
    "away_game_id",
    "home_date",
    "away_date",
    "date",
    "round",
    "home_club_goals",
    "away_club_goals",
    "diff_game_id",
    "diff_club_id",
    "diff_club_goals",
    "label",
    "is_shootout",
    "aggregate",
    "competition_id",
    "home_club_name",
    "away_club_name",
    "season",
    "stadium",
    "referee",
    "url",
    "home_club_formation",
    "away_club_formation",
    "home_club_position",
    "away_club_position",
    "own_manager_name",
    "opponent_manager_name",
    "home_manager_name",
    "away_manager_name",
}


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns while safely excluding leaks and identifiers."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    
    num_cols = []
    for c in df.columns:
        if c in EXCLUDE_COLUMNS or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
            
    return num_cols, cat_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for robust preprocessing."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


def build_ensemble_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """
    Construct high-performance multi-class ensemble pipeline:
    LightGBM + CatBoost + Random Forest with soft-voting.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=300,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    
    voting = VotingClassifier(
        estimators=[
            ("lgbm", lgbm),
            ("cat", cat),
            ("rf", rf),
        ],
        voting="soft",
        weights=[0.45, 0.40, 0.15],
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", voting),
    ])


def calibrate_pipeline(pipeline: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series | np.ndarray) -> CalibratedClassifierCV:
    """Calibrate probabilities using FrozenEstimator and sigmoid Platt scaling."""
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline),
            method="sigmoid",
            cv=3,
        )
    except (ImportError, Exception):
        calibrated = CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=3,
        )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def prob_to_decimal_odds(p: float) -> float:
    """Convert probability to European Decimal Odds (1 / P)."""
    p = max(0.001, min(0.999, float(p)))
    return round(1.0 / p, 2)


def prob_to_american_odds(p: float) -> str:
    """Convert probability to American Moneyline Odds."""
    p = max(0.001, min(0.999, float(p)))
    if p >= 0.5:
        odds = -int(round((p / (1.0 - p)) * 100))
        return f"{odds}"
    else:
        odds = int(round(((1.0 - p) / p) * 100))
        return f"+{odds}"


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate comprehensive predictive and calibration metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    loss = float(log_loss(y_true, y_prob))
    
    y_one_hot = np.zeros_like(y_prob)
    for i, val in enumerate(y_true):
        if not np.isnan(val):
            y_one_hot[i, int(val)] = 1.0
            
    brier = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        auc = float("nan")
        
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "log_loss": loss,
        "brier_score": brier,
        "roc_auc_ovr": auc,
        "draw_precision": float(report.get("1", {}).get("precision", 0.0)),
        "draw_recall": float(report.get("1", {}).get("recall", 0.0)),
        "draw_f1": float(report.get("1", {}).get("f1-score", 0.0)),
        "win_f1": float(report.get("2", {}).get("f1-score", 0.0)),
        "loss_f1": float(report.get("0", {}).get("f1-score", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
    }


def save_artifacts(pipeline: Pipeline, numeric_cols: list[str], categorical_cols: list[str], artifacts_dir: Path) -> None:
    """Save trained model pipeline and feature configuration."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "wc26_ensemble_model.joblib"
    joblib.dump(pipeline, model_path)
    
    feat_path = artifacts_dir / "feature_columns.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump({"numeric": numeric_cols, "categorical": categorical_cols}, f, indent=2)


def load_artifacts(artifacts_dir: Path) -> tuple[Pipeline, list[str], list[str]]:
    """Load model pipeline and feature configuration."""
    model_path = artifacts_dir / "wc26_ensemble_model.joblib"
    feat_path = artifacts_dir / "feature_columns.json"
    if not model_path.is_file() or not feat_path.is_file():
        raise FileNotFoundError(f"Model artifacts not found in {artifacts_dir}")
        
    pipeline = joblib.load(model_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta["numeric"], meta["categorical"]
