
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "extracted" / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

REPORT_DIR = BASE_DIR / "reports"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PREDICTIVE MAINTENANCE - SHAP EXPLAINABILITY")
    print("=" * 70)

    # Verify required files.
    for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # -----------------------------------------------------
    # LOAD MODEL AND METADATA
    # -----------------------------------------------------

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    features = metadata["features"]
    target = metadata["target"]

    print(f"\nModel: {metadata['model_name']}")
    print(f"Target: {target}")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df[features].copy()

    # Reproducible explainability sample.
    X_sample = X.sample(
        n=min(1000, len(X)),
        random_state=42,
    )

    print(f"Features: {len(features)}")
    print(f"Explainability sample: {len(X_sample)} records")

    # -----------------------------------------------------
    # EXTRACT PIPELINE COMPONENTS
    # -----------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Apply the exact preprocessing fitted during training.
    X_transformed = preprocessor.transform(X_sample)

    # Get transformed feature names.
    feature_names = preprocessor.get_feature_names_out()

    # XGBoost does not allow [, ], or < in feature names.
    safe_feature_names = [
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "_")
        .replace(">", "_")
        for name in feature_names
    ]

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=safe_feature_names,
        index=X_sample.index,
    )

    # -----------------------------------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------------------------------

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(
        X_transformed_df,
        check_additivity=False,
    )

    # -----------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = pd.DataFrame({
        "feature": safe_feature_names,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False,
            formatters={
                "mean_absolute_shap": "{:.6f}".format
            },
        )
    )

    # -----------------------------------------------------
    # SHAP SUMMARY PLOT
    # -----------------------------------------------------

    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        show=False,
        max_display=15,
    )

    plt.title(
        "SHAP Summary - Predictive Maintenance",
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        SCREENSHOT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # SHAP BAR PLOT
    # -----------------------------------------------------

    print("Generating SHAP importance bar plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        plot_type="bar",
        show=False,
        max_display=15,
    )

    plt.title(
        "SHAP Global Feature Importance",
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        SCREENSHOT_DIR / "shap_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPLAINABILITY OUTPUTS SAVED")
    print("=" * 70)

    print(" - reports/shap_feature_importance.csv")
    print(" - screenshots/shap_summary.png")
    print(" - screenshots/shap_importance.png")

    print("\nSHAP explainability completed successfully.")


if __name__ == "__main__":
    main()
