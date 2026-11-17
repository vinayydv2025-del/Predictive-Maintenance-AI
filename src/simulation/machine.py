from dataclasses import dataclass, field
from datetime import datetime, timezone
import random


@dataclass
class Machine:
    """
    Simulated industrial machine.

    Sensor readings and degradation are synthetic.
    """

    machine_id: str
    machine_type: str = "M"
    initial_health: float = 100.0
    random_seed: int = 42

    health: float = field(init=False)
    tool_wear: float = field(default=0.0, init=False)
    operating_hours: float = field(default=0.0, init=False)
    cycle: int = field(default=0, init=False)
    status: str = field(default="NORMAL", init=False)

    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        if self.machine_type not in {"L", "M", "H"}:
            raise ValueError("machine_type must be L, M, or H")

        if not 0 <= self.initial_health <= 100:
            raise ValueError("initial_health must be between 0 and 100")

        self.health = float(self.initial_health)
        self.rng = random.Random(self.random_seed)

    def _calculate_degradation(self):
        base_rate = {
            "L": 0.050,
            "M": 0.085,
            "H": 0.130,
        }[self.machine_type]
        wear_factor = 1.0 + self.tool_wear / 250.0
        degradation = base_rate * wear_factor

        self.health = max(0.0, self.health - degradation)

        self.tool_wear = min(
            250.0,
            self.tool_wear + self.rng.uniform(0.2, 0.8)
        )

    def _determine_status(self):
        if self.health <= 15:
            self.status = "CRITICAL"
        elif self.health <= 40:
            self.status = "DEGRADED"
        elif self.health <= 70:
            self.status = "WARNING"
        else:
            self.status = "NORMAL"

    def generate_sensor_reading(self):
        self.cycle += 1
        self.operating_hours += 0.1

        self._calculate_degradation()
        self._determine_status()

        degradation_ratio = (100.0 - self.health) / 100.0

        air_temperature = (
            298.0 + 8.0 * degradation_ratio + self.rng.gauss(0, 0.7)
        )

        process_temperature = (
            air_temperature + 10.0 + self.rng.gauss(0, 0.5)
        )

        rotational_speed = (
            1500.0 - 100.0 * degradation_ratio + self.rng.gauss(0, 25)
        )

        torque = (
            40.0 + 15.0 * degradation_ratio + self.rng.gauss(0, 2)
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "cycle": self.cycle,
            "operating_hours": round(self.operating_hours, 2),
            "health": round(self.health, 2),
            "tool_wear": round(self.tool_wear, 2),
            "air_temperature": round(air_temperature, 2),
            "process_temperature": round(process_temperature, 2),
            "rotational_speed": round(rotational_speed, 2),
            "torque": round(torque, 2),
            "status": self.status,
        }

    def perform_maintenance(self):
        self.health = min(100.0, self.health + 45.0)
        self.tool_wear = max(0.0, self.tool_wear - 100.0)

        self._determine_status()

        return {
            "machine_id": self.machine_id,
            "event": "MAINTENANCE_COMPLETED",
            "health_after": round(self.health, 2),
            "tool_wear_after": round(self.tool_wear, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
