"""Deterministic OPC-UA write-back for physical safety interlocks.

No LLM suggestion, confidence score, or agent output is accepted by this
module. A PLC write is possible only after direct sensor measurements pass
their physical-range checks and cross fixed safety thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Mapping

from asyncua import Client, ua


class SafetyAction(StrEnum):
    NONE = "none"
    THROTTLE = "throttle"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class SensorLimits:
    physical_min: float
    physical_max: float
    throttle_at: float
    emergency_stop_at: float

    def __post_init__(self) -> None:
        if not self.physical_min <= self.throttle_at <= self.emergency_stop_at <= self.physical_max:
            raise ValueError("limits must be ordered within physical bounds")


DEFAULT_LIMITS: dict[str, SensorLimits] = {
    "vibration": SensorLimits(0.0, 1_000.0, 7.1, 12.0),
    "thermal": SensorLimits(-60.0, 250.0, 85.0, 105.0),
    "acoustic": SensorLimits(0.0, 200.0, 105.0, 125.0),
}


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    reasons: tuple[str, ...]


class SafetyInterlockGate:
    """Evaluate direct physical measurements using fixed, auditable rules."""

    def __init__(self, limits: Mapping[str, SensorLimits] | None = None) -> None:
        self.limits = dict(limits or DEFAULT_LIMITS)

    def evaluate(self, measurements: Mapping[str, float]) -> SafetyDecision:
        expected = set(self.limits)
        missing = expected.difference(measurements)
        extra = set(measurements).difference(expected)
        if missing or extra:
            raise ValueError(f"measurements must contain exactly {sorted(expected)}")

        emergency, throttle = [], []
        for sensor, limits in self.limits.items():
            value = float(measurements[sensor])
            if not math.isfinite(value) or not limits.physical_min <= value <= limits.physical_max:
                raise ValueError(f"{sensor} measurement is outside validated physical bounds")
            if value >= limits.emergency_stop_at:
                emergency.append(f"{sensor}>={limits.emergency_stop_at:g}")
            elif value >= limits.throttle_at:
                throttle.append(f"{sensor}>={limits.throttle_at:g}")
        if emergency:
            return SafetyDecision(SafetyAction.EMERGENCY_STOP, tuple(emergency))
        if throttle:
            return SafetyDecision(SafetyAction.THROTTLE, tuple(throttle))
        return SafetyDecision(SafetyAction.NONE, ())


class OPCUASafetyClient:
    """Write hard safety actions to PLC tags after gate validation.

    Safe readings produce no control write. Restarts and manual reset remain a
    PLC/operator procedure; this service never clears an emergency stop.
    """

    def __init__(self, endpoint: str, *, emergency_stop_node: str,
                 speed_setpoint_node: str, throttle_speed: float,
                 gate: SafetyInterlockGate | None = None) -> None:
        if not math.isfinite(throttle_speed) or throttle_speed < 0:
            raise ValueError("throttle_speed must be a non-negative finite value")
        self.client = Client(url=endpoint)
        self.emergency_stop_node = emergency_stop_node
        self.speed_setpoint_node = speed_setpoint_node
        self.throttle_speed = throttle_speed
        self.gate = gate or SafetyInterlockGate()

    async def connect(self) -> None:
        await self.client.connect()

    async def disconnect(self) -> None:
        await self.client.disconnect()

    async def apply_measurements(self, measurements: Mapping[str, float]) -> SafetyDecision:
        """Gate, then write only an independently validated protective action."""
        decision = self.gate.evaluate(measurements)
        if decision.action is SafetyAction.EMERGENCY_STOP:
            node = self.client.get_node(self.emergency_stop_node)
            await node.write_value(ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))
        elif decision.action is SafetyAction.THROTTLE:
            node = self.client.get_node(self.speed_setpoint_node)
            await node.write_value(ua.DataValue(ua.Variant(self.throttle_speed, ua.VariantType.Double)))
        return decision
