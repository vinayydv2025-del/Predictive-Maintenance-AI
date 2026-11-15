
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


def predict_machine(machine_data, model, metadata):

    threshold = metadata["threshold"]

    # Predict failure probability using the complete pipeline.
    probability = float(model.predict_proba(machine_data)[0, 1])

    prediction = int(probability >= threshold)

    # Apply the same fitted preprocessing used during training.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_data = preprocessor.transform(machine_data)

    # Fast native XGBoost feature contributions.
    booster = classifier.get_booster()

    feature_names = preprocessor.get_feature_names_out()

    safe_feature_names = [
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "_")
        .replace(">", "_")
        for name in feature_names
    ]

    dmatrix = xgb.DMatrix(
        transformed_data,
        feature_names=safe_feature_names,
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
    )[0]

    explanation = pd.DataFrame({
        "feature": safe_feature_names,
        "shap_contribution": contributions[:-1],
    })

    explanation["absolute_contribution"] = (
        explanation["shap_contribution"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    return prediction, probability, threshold, explanation


def main():

    print("=" * 65)
    print("PREDICTIVE MAINTENANCE - MACHINE FAILURE PREDICTION")
    print("=" * 65)

    model, metadata = load_model()

    print("\nEnter machine readings:")

    values = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 200.0,
        "Type": "M",
    }

    machine_data = pd.DataFrame([values])

    prediction, probability, threshold, explanation = (
        predict_machine(machine_data, model, metadata)
    )

    print("\n" + "=" * 65)
    print("PREDICTION RESULTS")
    print("=" * 65)

    print(f"Model: {metadata['model_name']}")
    print(f"Prediction: {'FAILURE RISK' if prediction else 'NORMAL'}")
    print(f"Failure probability: {probability:.2%}")
    print(f"Decision threshold: {threshold:.4f}")

    print("\nMACHINE INPUTS")
    print(machine_data.to_string(index=False))

    print("\nFEATURE CONTRIBUTIONS")
    print("-" * 65)

    print(
        explanation[
            ["feature", "shap_contribution"]
        ].to_string(
            index=False,
            formatters={
                "shap_contribution": "{:+.6f}".format
            },
        )
    )

    print("\nPositive contribution = pushes toward failure.")
    print("Negative contribution = pushes toward normal.")

    print("\nIndividual prediction completed.")


if __name__ == "__main__":
    main()
