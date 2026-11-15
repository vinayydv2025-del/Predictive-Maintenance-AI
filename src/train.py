from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "extracted" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def select_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1_score": float(f1_values[best_index]),
    }


def main():
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE - VALIDATION-BASED MODEL SELECTION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # First split: 60% training, 40% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: divide temporary data equally into validation and test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("\nDataset split:")
    print(f"Training:   {len(X_train)} records ({len(X_train)/len(X):.0%})")
    print(f"Validation: {len(X_val)} records ({len(X_val)/len(X):.0%})")
    print(f"Test:       {len(X_test)} records ({len(X_test)/len(X):.0%})")

    print("\nFailure rates:")
    print(f"Training:   {y_train.mean():.2%}")
    print(f"Validation: {y_val.mean():.2%}")
    print(f"Test:       {y_test.mean():.2%}")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    validation_results = []
    fitted_models = {}

    for name, classifier in models.items():
        print("\n" + "-" * 70)
        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        result = {
            "model": name,
            "validation_roc_auc": roc_auc_score(y_val, val_probabilities),
            "validation_pr_auc": average_precision_score(
                y_val, val_probabilities
            ),
        }

        validation_results.append(result)
        fitted_models[name] = pipeline

        print(f"Validation ROC-AUC: {result['validation_roc_auc']:.4f}")
        print(f"Validation PR-AUC:  {result['validation_pr_auc']:.4f}")

    validation_df = pd.DataFrame(validation_results).sort_values(
        "validation_pr_auc", ascending=False
    )

    validation_df.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    best_model_name = validation_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)
    print(validation_df.round(4).to_string(index=False))
    print(f"\nSelected model: {best_model_name}")

    # Select the operating threshold using validation data only.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    threshold_result = select_f1_threshold(y_val, val_probabilities)
    selected_threshold = threshold_result["threshold"]

    print("\nSelected validation threshold:")
    print(f"Threshold: {selected_threshold:.6f}")
    print(f"Validation precision: {threshold_result['precision']:.4f}")
    print(f"Validation recall: {threshold_result['recall']:.4f}")
    print(f"Validation F1: {threshold_result['f1_score']:.4f}")

    # Final evaluation: test data is used only here.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    test_metrics = {
        "model": best_model_name,
        "threshold": float(selected_threshold),
        "precision": float(precision_score(
            y_test, test_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, test_predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, test_predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "pr_auc": float(average_precision_score(
            y_test, test_probabilities
        )),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(matrix)

    print("\nFinal metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Save the selected pipeline and its operating threshold.
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    metadata = {
        "model_name": best_model_name,
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": float(selected_threshold),
        "threshold_selection": "Maximum validation F1",
        "selection_metric": "Validation PR-AUC",
        "random_state": RANDOM_STATE,
        "split": {
            "training": 0.60,
            "validation": 0.20,
            "test": 0.20,
        },
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    with open(
        REPORT_DIR / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_metrics, file, indent=4)

    print("\nSaved artifacts:")
    print(" - reports/validation_model_comparison.csv")
    print(" - reports/final_test_metrics.json")
    print(" - models/best_model.joblib")
    print(" - models/model_metadata.json")


if __name__ == "__main__":
    main()
