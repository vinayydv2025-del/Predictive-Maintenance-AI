
import random


class FaultInjector:
    """
    Inject controlled synthetic faults and sensor noise.

    Supported equipment faults:
        OVERHEATING
        OVERLOAD
        SPEED_ANOMALY

    Supports individual and simultaneous fault injection.

    Sensor noise is tracked separately from equipment faults.

    These are synthetic simulation mechanisms, not validated
    physical failure models.
    """

    SUPPORTED_FAULTS = {
        "OVERHEATING",
        "OVERLOAD",
        "SPEED_ANOMALY"
    }

    SENSOR_NOISE_SCALES = {
        "air_temperature": 0.7,
        "process_temperature": 0.7,
        "rotational_speed": 25.0,
        "torque": 2.0,
        "tool_wear": 1.0,
        "operating_hours": 0.1
    }

    def __init__(self, random_seed=42):
        self.rng = random.Random(random_seed)

    # -----------------------------------------------------
    # INDIVIDUAL FAULT INJECTION
    # -----------------------------------------------------

    def _apply_fault(self, reading, fault_type):

        if fault_type not in self.SUPPORTED_FAULTS:
            raise ValueError(
                f"Unsupported fault type: {fault_type}"
            )

        result = reading.copy()

        if fault_type == "OVERHEATING":

            increase = self.rng.uniform(12.0, 20.0)

            result["air_temperature"] = round(
                result["air_temperature"] + increase, 2
            )

            result["process_temperature"] = round(
                result["process_temperature"] + increase + 3.0, 2
            )

        elif fault_type == "OVERLOAD":

            increase = self.rng.uniform(20.0, 35.0)

            result["torque"] = round(
                result["torque"] + increase, 2
            )

        elif fault_type == "SPEED_ANOMALY":

            reduction = self.rng.uniform(250.0, 450.0)

            result["rotational_speed"] = round(
                max(
                    0.0,
                    result["rotational_speed"] - reduction
                ),
                2
            )

        return result

    # -----------------------------------------------------
    # SINGLE FAULT
    # -----------------------------------------------------

    def inject(self, reading, fault_type):
        """
        Inject one fault.

        Preserves the legacy anomaly_type field so existing
        single-fault dataset generation remains compatible.
        """

        result = self._apply_fault(
            reading,
            fault_type
        )

        existing_faults = list(
            result.get("fault_types", [])
        )

        if fault_type not in existing_faults:
            existing_faults.append(fault_type)

        result["is_anomaly"] = True
        result["anomaly_type"] = fault_type
        result["fault_types"] = existing_faults

        return result

    # -----------------------------------------------------
    # MULTIPLE SIMULTANEOUS FAULTS
    # -----------------------------------------------------

    def inject_multiple(self, reading, fault_types):
        """
        Inject multiple independent synthetic faults.

        Example:
            ["OVERHEATING", "OVERLOAD"]

        Returns independent fault labels in fault_types.

        The legacy anomaly_type field remains available:
            One fault: its name
            Multiple faults: MULTIPLE_FAULTS
        """

        if not fault_types:
            raise ValueError(
                "At least one fault must be specified."
            )

        if len(fault_types) != len(set(fault_types)):
            raise ValueError(
                "Duplicate fault types are not allowed."
            )

        unsupported = (
            set(fault_types) - self.SUPPORTED_FAULTS
        )

        if unsupported:
            raise ValueError(
                f"Unsupported faults: {sorted(unsupported)}"
            )

        result = reading.copy()

        for fault_type in fault_types:
            result = self._apply_fault(
                result,
                fault_type
            )

        result["is_anomaly"] = True
        result["fault_types"] = list(fault_types)

        result["anomaly_type"] = (
            fault_types[0]
            if len(fault_types) == 1
            else "MULTIPLE_FAULTS"
        )

        return result

    # -----------------------------------------------------
    # SENSOR NOISE
    # -----------------------------------------------------

    def inject_sensor_noise(
        self,
        reading,
        noise_multiplier=1.0
    ):
        """
        Add synthetic Gaussian sensor noise.

        Sensor noise is not automatically an equipment fault.

        sensor_quality_issue is tracked separately from
        is_anomaly and fault_types.
        """

        if noise_multiplier < 0:
            raise ValueError(
                "noise_multiplier cannot be negative."
            )

        result = reading.copy()

        for feature, base_scale in self.SENSOR_NOISE_SCALES.items():

            if feature not in result:
                continue

            noise = self.rng.gauss(
                0,
                base_scale * noise_multiplier
            )

            result[feature] = round(
                float(result[feature]) + noise,
                4
            )

        result["sensor_quality_issue"] = True
        result["sensor_noise_multiplier"] = float(
            noise_multiplier
        )

        result.setdefault("is_anomaly", False)
        result.setdefault("fault_types", [])
        result.setdefault("anomaly_type", None)

        return result

    # -----------------------------------------------------
    # NORMAL READING
    # -----------------------------------------------------

    def normal_reading(self, reading):
        """Label an unchanged reading as normal."""

        result = reading.copy()

        result["is_anomaly"] = False
        result["anomaly_type"] = None
        result["fault_types"] = []
        result["sensor_quality_issue"] = False

        return result
