
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fleet_intelligence.risk_ranking import rank_machines_by_risk


def create_machine(
    machine_id,
    diagnosis,
    severity,
    faults=None,
    sensor_warning=False
):
    return {
        "machine_id": machine_id,
        "final_diagnosis": {
            "diagnosis": diagnosis,
            "severity": severity,
            "detected_faults": faults or [],
            "supporting_alert_count": len(faults or []),
            "diagnostic_reliability": (
                "COMPROMISED"
                if sensor_warning
                else "ACTIONABLE_WITH_INVESTIGATION"
                if severity != "NORMAL"
                else "NO_ALERT"
            ),
            "maintenance_action": (
                "VERIFY_SENSORS_BEFORE_FINAL_ASSESSMENT"
                if sensor_warning
                else "CONTINUE_ROUTINE_MONITORING"
                if severity == "NORMAL"
                else "INVESTIGATE_EQUIPMENT"
            ),
            "sensor_quality": {
                "status": (
                    "SENSOR_QUALITY_WARNING"
                    if sensor_warning
                    else "NOT_ASSESSED"
                )
            }
        }
    }


# ---------------------------------------------------------
# CREATE SYNTHETIC FLEET
# ---------------------------------------------------------

machine_results = [
    create_machine("MACHINE-001", "NORMAL_OPERATION", "NORMAL"),
    create_machine("MACHINE-002", "NORMAL_OPERATION", "NORMAL"),
    create_machine("MACHINE-003", "NORMAL_OPERATION", "NORMAL"),

    create_machine(
        "MACHINE-004",
        "OVERHEATING",
        "WARNING",
        ["OVERHEATING"]
    ),

    create_machine(
        "MACHINE-005",
        "OVERLOAD",
        "WARNING",
        ["OVERLOAD"]
    ),

    create_machine(
        "MACHINE-006",
        "MULTIPLE_FAULTS",
        "HIGH",
        ["OVERHEATING", "OVERLOAD"]
    ),

    create_machine(
        "MACHINE-007",
        "MULTIPLE_FAULTS",
        "CRITICAL",
        ["OVERHEATING", "OVERLOAD", "SPEED_ANOMALY"]
    ),

    create_machine(
        "MACHINE-008",
        "MULTIPLE_FAULTS",
        "CRITICAL",
        ["OVERHEATING", "OVERLOAD", "SPEED_ANOMALY"],
        sensor_warning=True
    )
]


# ---------------------------------------------------------
# RUN RISK RANKING
# ---------------------------------------------------------

ranked_machines = rank_machines_by_risk(machine_results)


print("=" * 85)
print("FLEET MACHINE RISK RANKING")
print("=" * 85)

for machine in ranked_machines:
    print(
        f"Rank {machine['risk_rank']}: "
        f"{machine['machine_id']} | "
        f"Severity={machine['severity']} | "
        f"Diagnosis={machine['diagnosis']} | "
        f"Faults={machine['detected_fault_count']} | "
        f"Reliability={machine['diagnostic_reliability']} | "
        f"Score={machine['ranking_score']} | "
        f"Action={machine['maintenance_action']}"
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

rank_by_id = {
    machine["machine_id"]: machine["risk_rank"]
    for machine in ranked_machines
}

checks = {
    "All eight machines ranked":
        len(ranked_machines) == 8,

    "Ranks are unique":
        len(set(rank_by_id.values())) == 8,

    "Ranks are sequential":
        sorted(rank_by_id.values()) == list(range(1, 9)),

    "Critical machines rank above high-risk machines":
        max(
            rank_by_id["MACHINE-007"],
            rank_by_id["MACHINE-008"]
        ) < rank_by_id["MACHINE-006"],

    "High-risk machine ranks above warning machines":
        rank_by_id["MACHINE-006"] < rank_by_id["MACHINE-004"]
        and rank_by_id["MACHINE-006"] < rank_by_id["MACHINE-005"],

    "Sensor-compromised critical machine ranks first":
        rank_by_id["MACHINE-008"] == 1,

    "Normal machines rank below warning machines":
        min(
            rank_by_id["MACHINE-001"],
            rank_by_id["MACHINE-002"],
            rank_by_id["MACHINE-003"]
        ) > max(
            rank_by_id["MACHINE-004"],
            rank_by_id["MACHINE-005"]
        ),

    "Sensor warning is retained in ranked result":
        next(
            machine for machine in ranked_machines
            if machine["machine_id"] == "MACHINE-008"
        )["sensor_quality"] == "SENSOR_QUALITY_WARNING"
}


print("\n" + "=" * 85)
print("RISK RANKING VALIDATION")
print("=" * 85)

for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


failed = [
    name for name, passed in checks.items()
    if not passed
]

print(f"\nPassed: {len(checks) - len(failed)}/{len(checks)}")

if failed:
    print("\nFailed checks:")
    for name in failed:
        print(name)
else:
    print("\nAll fleet risk ranking tests passed.")
