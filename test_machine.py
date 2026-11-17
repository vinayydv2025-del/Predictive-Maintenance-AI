from src.simulation.machine import Machine

machine = Machine(
    machine_id="MACHINE-001",
    machine_type="M",
    random_seed=42
)

print("=" * 65)
print("INDUSTRIAL DIGITAL TWIN - MACHINE SIMULATION")
print("=" * 65)

for _ in range(5):
    reading = machine.generate_sensor_reading()

    print(
        f"Cycle: {reading['cycle']:>3} | "
        f"Health: {reading['health']:>6}% | "
        f"Wear: {reading['tool_wear']:>6} | "
        f"Torque: {reading['torque']:>7} | "
        f"Status: {reading['status']}"
    )

print("\n===== MAINTENANCE TEST =====")
print(machine.perform_maintenance())
