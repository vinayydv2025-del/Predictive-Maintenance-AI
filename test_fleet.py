from src.simulation.machine import Machine
from src.simulation.fleet import FleetSimulator


fleet = FleetSimulator()

machine_configs = [
    ("MACHINE-001", "L", 42),
    ("MACHINE-002", "M", 43),
    ("MACHINE-003", "H", 44),
    ("MACHINE-004", "M", 45),
    ("MACHINE-005", "L", 46),
]

for machine_id, machine_type, seed in machine_configs:
    machine = Machine(
        machine_id=machine_id,
        machine_type=machine_type,
        random_seed=seed,
    )

    fleet.add_machine(machine)


print("=" * 75)
print("INDUSTRIAL FLEET SIMULATOR")
print("=" * 75)

for cycle in range(1, 4):
    print(f"\n===== FLEET CYCLE {cycle} =====")

    readings = fleet.step()

    for reading in readings:
        print(
            f"{reading['machine_id']} | "
            f"Type: {reading['machine_type']} | "
            f"Health: {reading['health']:>6}% | "
            f"Wear: {reading['tool_wear']:>6} | "
            f"Status: {reading['status']}"
        )


print("\n===== FLEET HEALTH SUMMARY =====")

summary = fleet.get_fleet_status()

print(f"Total machines: {summary['total_machines']}")
print(f"Average fleet health: {summary['average_health']}%")

for status, count in summary["status_counts"].items():
    print(f"{status}: {count}")


print("\n===== MACHINE-LEVEL SUMMARY =====")

for machine in summary["machines"]:
    print(
        f"{machine['machine_id']} | "
        f"Health: {machine['health']}% | "
        f"Status: {machine['status']}"
    )


print("\n===== MAINTENANCE TEST =====")

result = fleet.perform_maintenance("MACHINE-003")
print(result)
