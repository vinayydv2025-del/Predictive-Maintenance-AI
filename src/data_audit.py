from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "extracted" / "ai4i2020.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("=" * 60)
print("PREDICTIVE MAINTENANCE - DATA AUDIT")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\nTarget distribution:")
print(df["Machine failure"].value_counts())

print("\nTarget percentages:")
print(
    (df["Machine failure"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nProduct type failure rates:")
print(
    df.groupby("Type")["Machine failure"]
    .agg(["count", "sum", "mean"])
    .rename(columns={"sum": "failures", "mean": "failure_rate"})
)

print("\nFailure mode counts:")
print(df[failure_modes].sum())

print("\nSensor statistics:")
print(df[sensor_columns].describe().round(2))

report_path = REPORT_DIR / "data_quality_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("PREDICTIVE MAINTENANCE DATA QUALITY REPORT\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Missing values: {df.isnull().sum().sum()}\n")
    file.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")
    file.write("Target distribution:\n")
    file.write(df["Machine failure"].value_counts().to_string())
    file.write("\n\nFailure modes:\n")
    file.write(df[failure_modes].sum().to_string())
    file.write("\n\nSensor statistics:\n")
    file.write(df[sensor_columns].describe().round(2).to_string())

print(f"\nReport saved to: {report_path}")
