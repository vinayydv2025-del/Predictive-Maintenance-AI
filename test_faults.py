from src.simulation.machine import Machine
from src.simulation.faults import FaultInjector


machine = Machine(
    machine_id="MACHINE-FAULT-001",
    machine_type="M",
    random_seed=42
)

injector = FaultInjector(random_seed=100)

normal_reading = machine.generate_sensor_reading()

print("=" * 75)
print("INDUSTRIAL FAULT INJECTION TEST")
print("=" * 75)

print("\n===== NORMAL SENSOR READING =====")
print(
    f"Temperature: {normal_reading['process_temperature']} K | "
    f"Torque: {normal_reading['torque']} Nm | "
    f"Speed: {normal_reading['rotational_speed']} rpm"
)

fault_types = [
    "OVERHEATING",
    "OVERLOAD",
    "SPEED_ANOMALY",
]

for fault_type in fault_types:

    faulty_reading = injector.inject(
        normal_reading,
        fault_type
    )

    print(f"\n===== {fault_type} =====")

    print(
        f"Air temperature: {faulty_reading['air_temperature']} K"
    )

    print(
        f"Process temperature: "
        f"{faulty_reading['process_temperature']} K"
    )

    print(
        f"Torque: {faulty_reading['torque']} Nm"
    )

    print(
        f"Rotational speed: "
        f"{faulty_reading['rotational_speed']} rpm"
    )

    print(
        f"Anomaly detected in simulation: "
        f"{faulty_reading['is_anomaly']}"
    )

    print(
        f"Anomaly type: {faulty_reading['anomaly_type']}"
    )


print("\n===== NORMAL READING LABEL TEST =====")

labelled_normal = injector.normal_reading(normal_reading)

print(f"Is anomaly: {labelled_normal['is_anomaly']}")
print(f"Anomaly type: {labelled_normal['anomaly_type']}")
