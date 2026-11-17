from src.simulation.machine import Machine


class FleetSimulator:
    """
    Manages multiple simulated industrial machines.

    All machine readings and health values are synthetic.
    """

    def __init__(self):
        self.machines = {}

    def add_machine(self, machine):
        """Register a machine in the fleet."""

        if machine.machine_id in self.machines:
            raise ValueError(
                f"Machine {machine.machine_id} already exists."
            )

        self.machines[machine.machine_id] = machine

    def step(self):
        """Advance every registered machine by one operating cycle."""

        readings = []

        for machine in self.machines.values():
            readings.append(machine.generate_sensor_reading())

        return readings

    def get_fleet_status(self):
        """Generate a summary of the current fleet condition."""

        machines = list(self.machines.values())

        if not machines:
            return {
                "total_machines": 0,
                "average_health": 0,
                "status_counts": {},
                "machines": [],
            }

        status_counts = {
            "NORMAL": 0,
            "WARNING": 0,
            "DEGRADED": 0,
            "CRITICAL": 0,
        }

        machine_details = []

        for machine in machines:
            status_counts[machine.status] += 1

            machine_details.append({
                "machine_id": machine.machine_id,
                "machine_type": machine.machine_type,
                "health": round(machine.health, 2),
                "tool_wear": round(machine.tool_wear, 2),
                "operating_hours": round(machine.operating_hours, 2),
                "status": machine.status,
            })

        average_health = sum(
            machine.health for machine in machines
        ) / len(machines)

        return {
            "total_machines": len(machines),
            "average_health": round(average_health, 2),
            "status_counts": status_counts,
            "machines": machine_details,
        }

    def perform_maintenance(self, machine_id):
        """Perform maintenance on a selected machine."""

        if machine_id not in self.machines:
            raise KeyError(f"Machine {machine_id} not found.")

        return self.machines[machine_id].perform_maintenance()
