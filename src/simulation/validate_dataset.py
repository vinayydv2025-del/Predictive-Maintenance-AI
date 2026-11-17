import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/simulated/sensor_readings.csv")
SUMMARY_PATH = Path("data/simulated/dataset_summary.json")

df = pd.read_csv(DATA_PATH)

print("=" * 75)
print("INDUSTRIAL SENSOR DATASET - QUALITY VALIDATION")
print("=" * 75)

print("\n===== DATASET DIMENSIONS =====")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Required columns must never contain missing values.
required_columns = [
    column for column in df.columns
    if column != "anomaly_type"
]

print("\n===== MISSING VALUE ANALYSIS =====")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n===== DUPLICATE ANALYSIS =====")
duplicate_rows = int(df.duplicated().sum())
duplicate_keys = int(
    df.duplicated(subset=["machine_id", "cycle"]).sum()
)

print(f"Duplicate complete rows: {duplicate_rows}")
print(f"Duplicate machine-cycle combinations: {duplicate_keys}")

print("\n===== ANOMALY DISTRIBUTION =====")
print(df["is_anomaly"].value_counts().sort_index())

print("\n===== ANOMALY TYPE DISTRIBUTION =====")
print(df["anomaly_type"].value_counts(dropna=False))

print("\n===== LABEL CONSISTENCY =====")

invalid_normal = int(
    (
        (df["is_anomaly"] == 0)
        & df["anomaly_type"].notna()
    ).sum()
)

invalid_anomaly = int(
    (
        (df["is_anomaly"] == 1)
        & df["anomaly_type"].isna()
    ).sum()
)

print(f"Normal rows with fault labels: {invalid_normal}")
print(f"Anomaly rows without fault labels: {invalid_anomaly}")

print("\n===== MACHINE DISTRIBUTION =====")
print(df.groupby("machine_id").size().describe())

print("\n===== MACHINE TYPE DISTRIBUTION =====")
print(df.groupby("machine_type")["machine_id"].nunique())

sensor_columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "health",
]

print("\n===== SENSOR STATISTICS =====")
print(df[sensor_columns].describe().round(2))

print("\n===== HEALTH STATUS DISTRIBUTION =====")
print(df["status"].value_counts())

print("\n===== HEALTH DISTRIBUTION BY STATUS =====")
print(
    df.groupby("status")["health"]
    .agg(["min", "mean", "max"])
    .round(2)
)

print("\n===== TIMESTAMP VALIDATION =====")

timestamps = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

print(f"Invalid timestamps: {timestamps.isna().sum()}")
print(f"Earliest timestamp: {timestamps.min()}")
print(f"Latest timestamp: {timestamps.max()}")

# Convert every check to a native Python bool for JSON serialization.
checks = {
    "correct_row_count": bool(len(df) == 10000),

    "no_missing_required_fields": bool(
        not df[required_columns].isnull().any().any()
    ),

    "anomaly_type_missing_only_for_normal": bool(
        invalid_normal == 0
    ),

    "anomaly_type_present_for_anomalies": bool(
        invalid_anomaly == 0
    ),

    "no_duplicate_rows": bool(duplicate_rows == 0),

    "no_duplicate_machine_cycles": bool(duplicate_keys == 0),

    "valid_anomaly_values": bool(
        set(df["is_anomaly"].unique()).issubset({0, 1})
    ),

    "valid_health_range": bool(
        df["health"].between(0, 100).all()
    ),

    "valid_status_values": bool(
        set(df["status"].unique()).issubset(
            {"NORMAL", "WARNING", "DEGRADED", "CRITICAL"}
        )
    ),

    "valid_timestamps": bool(not timestamps.isna().any()),

    "valid_fault_types": bool(
        set(df["anomaly_type"].dropna().unique()).issubset(
            {"OVERHEATING", "OVERLOAD", "SPEED_ANOMALY"}
        )
    ),
}

print("\n===== VALIDATION CHECKS =====")

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

report = {
    "dataset": str(DATA_PATH),
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_anomaly_type_values": int(
        df["anomaly_type"].isna().sum()
    ),
    "checks": checks,
    "all_checks_passed": bool(all(checks.values())),
}

with SUMMARY_PATH.open("r", encoding="utf-8") as file:
    report["generation_summary"] = json.load(file)

report_path = Path("reports/synthetic_dataset_validation.json")
report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n===== FINAL RESULT =====")

if all(checks.values()):
    print("ALL VALIDATION CHECKS PASSED")
else:
    print("SOME CHECKS FAILED - REVIEW REQUIRED")

print(f"Validation report saved: {report_path}")