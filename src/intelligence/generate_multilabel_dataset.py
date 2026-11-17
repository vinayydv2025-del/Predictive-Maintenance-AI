
import json
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.simulation.faults import FaultInjector


INPUT_PATH = ROOT / "data" / "simulated" / "sensor_readings.csv"
OUTPUT_PATH = ROOT / "data" / "simulated" / "multi_label_fault_dataset.csv"
REPORT_PATH = ROOT / "reports" / "multilabel_dataset_summary.json"

FAULT_TYPES = [
    "OVERHEATING",
    "OVERLOAD",
    "SPEED_ANOMALY"
]

# Same machine-held-out split used in the earlier
# failure-mode classifier experiment.

TRAIN_MACHINES = [
    "MACHINE-001", "MACHINE-004", "MACHINE-006",
    "MACHINE-007", "MACHINE-008", "MACHINE-010",
    "MACHINE-011", "MACHINE-013", "MACHINE-015",
    "MACHINE-016", "MACHINE-017", "MACHINE-020"
]

CALIBRATION_MACHINES = [
    "MACHINE-003", "MACHINE-012", "MACHINE-019"
]

TEST_MACHINES = [
    "MACHINE-002", "MACHINE-005", "MACHINE-009",
    "MACHINE-014", "MACHINE-018"
]

FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
    "machine_type"
]


def get_machine_split(machine_id):

    if machine_id in TRAIN_MACHINES:
        return "TRAIN"

    if machine_id in CALIBRATION_MACHINES:
        return "CALIBRATION"

    if machine_id in TEST_MACHINES:
        return "TEST"

    raise ValueError(
        f"Unknown machine in split configuration: {machine_id}"
    )


def main():

    print("=" * 75)
    print("MULTI-LABEL INDUSTRIAL FAULT DATASET GENERATOR")
    print("=" * 75)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\n===== SOURCE DATASET =====")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    required = [
        "machine_id",
        "machine_type",
        "cycle",
        "timestamp",
        "is_anomaly"
    ] + FEATURES[:-1]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required source columns: {missing}"
        )

    # Use only clean normal readings as the starting point.
    normal_df = df[
        df["is_anomaly"].astype(int) == 0
    ].copy()

    if normal_df.empty:
        raise ValueError(
            "No normal source observations were found."
        )

    print("Clean starting observations:", len(normal_df))

    # Verify machine split coverage.
    all_machines = set(df["machine_id"].astype(str))

    split_machines = (
        set(TRAIN_MACHINES)
        | set(CALIBRATION_MACHINES)
        | set(TEST_MACHINES)
    )

    if all_machines != split_machines:
        raise ValueError(
            "Machine split does not match source machines."
        )

    if (
        set(TRAIN_MACHINES) & set(CALIBRATION_MACHINES)
        or set(TRAIN_MACHINES) & set(TEST_MACHINES)
        or set(CALIBRATION_MACHINES) & set(TEST_MACHINES)
    ):
        raise ValueError(
            "Machine overlap detected in split configuration."
        )

    # Generate all 8 combinations:
    # 1 normal + 3 individual + 3 double + 1 triple.
    scenarios = [()]

    for number_of_faults in range(1, len(FAULT_TYPES) + 1):
        scenarios.extend(
            combinations(FAULT_TYPES, number_of_faults)
        )

    print("\nScenarios per source observation:", len(scenarios))

    injector = FaultInjector(random_seed=2026)

    records = []

    print("\n===== GENERATING SCENARIOS =====")

    for row_number, (_, row) in enumerate(
        normal_df.iterrows(),
        start=1
    ):

        base = {
            "machine_id": str(row["machine_id"]),
            "machine_type": str(row["machine_type"]),
            "air_temperature": float(row["air_temperature"]),
            "process_temperature": float(row["process_temperature"]),
            "rotational_speed": float(row["rotational_speed"]),
            "torque": float(row["torque"]),
            "tool_wear": float(row["tool_wear"]),
            "operating_hours": float(row["operating_hours"])
        }

        for fault_combination in scenarios:

            if fault_combination:

                simulated = injector.inject_multiple(
                    base,
                    list(fault_combination)
                )

            else:

                simulated = injector.normal_reading(base)

            record = {
                "machine_id": base["machine_id"],
                "machine_type": simulated["machine_type"],
                "cycle": int(row["cycle"]),
                "timestamp": str(row["timestamp"]),
                "split": get_machine_split(base["machine_id"]),
                "scenario_type": (
                    "NORMAL"
                    if not fault_combination
                    else (
                        "SINGLE_FAULT"
                        if len(fault_combination) == 1
                        else (
                            "DOUBLE_FAULT"
                            if len(fault_combination) == 2
                            else "TRIPLE_FAULT"
                        )
                    )
                ),
                "fault_combination": (
                    "+".join(fault_combination)
                    if fault_combination
                    else "NORMAL"
                ),
                "air_temperature": simulated["air_temperature"],
                "process_temperature": simulated["process_temperature"],
                "rotational_speed": simulated["rotational_speed"],
                "torque": simulated["torque"],
                "tool_wear": simulated["tool_wear"],
                "operating_hours": simulated["operating_hours"],
                "fault_overheating": int(
                    "OVERHEATING" in fault_combination
                ),
                "fault_overload": int(
                    "OVERLOAD" in fault_combination
                ),
                "fault_speed_anomaly": int(
                    "SPEED_ANOMALY" in fault_combination
                ),
                "is_anomaly": int(bool(fault_combination)),
                "sensor_quality_issue": 0
            }

            records.append(record)

        if row_number % 1000 == 0:
            print(
                f"Processed {row_number:,} "
                f"/ {len(normal_df):,} source observations"
            )

    result = pd.DataFrame(records)

    # Dataset validation.
    target_columns = [
        "fault_overheating",
        "fault_overload",
        "fault_speed_anomaly"
    ]

    expected_rows = len(normal_df) * len(scenarios)

    checks = {
        "expected_row_count": len(result) == expected_rows,
        "no_missing_values": not result.isnull().any().any(),
        "no_duplicate_rows": not result.duplicated().any(),
        "valid_binary_targets": all(
            set(result[column].unique()).issubset({0, 1})
            for column in target_columns
        ),
        "normal_labels_consistent": (
            result.loc[
                result["scenario_type"] == "NORMAL",
                target_columns
            ].sum().sum() == 0
        ),
        "anomaly_labels_consistent": (
            (
                result[target_columns].sum(axis=1) > 0
            ).astype(int).equals(result["is_anomaly"])
        ),
        "machine_split_valid": result["split"].notna().all()
    }

    # Confirm no machine crosses split boundaries.
    split_overlap = False

    for machine_id, group in result.groupby("machine_id"):
        if group["split"].nunique() != 1:
            split_overlap = True

    checks["machine_separation"] = not split_overlap

    if not all(checks.values()):
        raise ValueError(
            f"Dataset validation failed: {checks}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "dataset": "Synthetic multi-label industrial fault dataset",
        "source_dataset": str(INPUT_PATH),
        "source_normal_observations": len(normal_df),
        "scenarios_per_observation": len(scenarios),
        "total_observations": len(result),
        "features": FEATURES,
        "targets": target_columns,
        "split_counts": {
            str(key): int(value)
            for key, value in result["split"].value_counts().items()
        },
        "machine_counts": {
            str(key): int(value)
            for key, value in result.groupby("split")["machine_id"].nunique().items()
        },
        "scenario_counts": {
            str(key): int(value)
            for key, value in result["scenario_type"].value_counts().items()
        },
        "fault_positive_counts": {
            column: int(result[column].sum())
            for column in target_columns
        },
        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },
        "limitations": [
            "All fault signals are synthetically injected.",
            "The simulator does not represent validated physical failure mechanisms.",
            "Training and test observations are derived from the same simulator design.",
            "Machine-held-out testing evaluates unseen simulated machines, not real industrial generalization.",
            "Sensor noise is not included in this equipment-fault dataset."
        ]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n===== DATASET SUMMARY =====")
    print("Total observations:", len(result))
    print("\nScenario distribution:")
    print(result["scenario_type"].value_counts())

    print("\nFault positive counts:")
    for column in target_columns:
        print(column, result[column].sum())

    print("\nMachine split:")
    print(result.groupby("split")["machine_id"].nunique())

    print("\n===== VALIDATION =====")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")

    print("\nDataset saved:", OUTPUT_PATH)
    print("Summary saved:", REPORT_PATH)

    print("\nMULTI-LABEL DATASET GENERATION COMPLETED")


if __name__ == "__main__":
    main()
