"""
ACA Tools — Actuator & Phytosanitary Treatment Tools
=====================================================

Provides simulated actuator adapters for closed-loop physical interventions:
    - ``IrrigationControlTool``: Adjusts microclimate soil moisture and canopy
      irrigation regimes (e.g. decrease, increase, stop, maintain).
    - ``TreatmentAlertTool``: Dispatches targeted phytosanitary alerts and
      agronomic remediation directives.

Architectural Guarantees:
    - Inherits from ``BaseTool`` with declarative ``ToolSchema``.
    - Enforces input validation and structured ``ToolResult`` returns.
    - Zero global singletons; constructor-injected configurations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aca.logging_config import get_logger
from aca.tools.base_tool import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

logger = get_logger("tools.actuators")


class IrrigationControlTool(BaseTool):
    """
    Simulated actuator tool for controlling greenhouse / plot irrigation valves.

    Accepted Actions:
        - ``decrease``: Reduce irrigation volume/frequency to lower leaf wetness.
        - ``increase``: Increase irrigation for drought-stressed or dehydrated crops.
        - ``stop``: Immediately halt irrigation (emergency fungal/bacterial containment).
        - ``start``: Activate standard scheduled irrigation cycle.
        - ``maintain``: Keep current baseline irrigation schedule unchanged.
    """

    ALLOWED_ACTIONS = {"decrease", "increase", "stop", "start", "maintain"}

    @property
    def schema(self) -> ToolSchema:
        """Declarative parameter specification for the irrigation tool."""
        return ToolSchema(
            name="irrigation_control",
            description="Control irrigation valves, schedule intervals, and flow rates for target greenhouse zones",
            parameters=[
                ToolParameter(
                    name="action",
                    description="Irrigation command: decrease, increase, stop, start, maintain",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="zone",
                    description="Target farm or greenhouse zone identifier",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="amount_litres",
                    description="Target volume in litres if applicable",
                    param_type="float",
                    required=False,
                    default=0.0,
                ),
                ToolParameter(
                    name="reason",
                    description="Agronomic justification for the valve state change",
                    param_type="str",
                    required=False,
                    default="",
                ),
            ],
            returns="Structured execution status with valve state and confirmation timestamp",
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute irrigation command on target zone valve actuator.

        Args:
            **kwargs: Must contain ``action`` and ``zone``.

        Returns:
            ``ToolResult`` with execution confirmation and metadata.
        """
        action = str(kwargs.get("action", "")).lower().strip()
        zone = str(kwargs.get("zone", "")).strip()
        amount = float(kwargs.get("amount_litres", 0.0))
        reason = str(kwargs.get("reason", ""))

        if not action or action not in self.ALLOWED_ACTIONS:
            return ToolResult(
                success=False,
                error=f"Invalid action '{action}'. Allowed: {sorted(self.ALLOWED_ACTIONS)}",
            )
        if not zone:
            return ToolResult(
                success=False,
                error="Missing required parameter 'zone'",
            )

        ts = datetime.now(timezone.utc).isoformat()
        execution_id = f"irr_cmd_{uuid.uuid4().hex[:8]}"

        logger.info(
            "IrrigationControlTool executed '%s' on zone '%s' (amount=%.1fL, reason='%s')",
            action,
            zone,
            amount,
            reason,
        )

        return ToolResult(
            success=True,
            data={
                "tool": "irrigation_control",
                "execution_id": execution_id,
                "action": action,
                "zone": zone,
                "amount_litres": amount,
                "reason": reason,
                "status": "EXECUTED",
                "timestamp": ts,
            },
            metadata={"valve_state": "ACTIVE" if action in ("start", "increase") else "THROTTLED" if action == "decrease" else "CLOSED"},
        )


class TreatmentAlertTool(BaseTool):
    """
    Simulated actuator/alerting tool for dispatching phytosanitary treatment directives.

    Directs chemical, biological, or sanitation interventions to greenhouse operators
    or automated dosing systems.
    """

    ALLOWED_URGENCIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    @property
    def schema(self) -> ToolSchema:
        """Declarative parameter specification for the treatment alert tool."""
        return ToolSchema(
            name="treatment_alert",
            description="Dispatch phytosanitary treatment alerts and chemical/biological intervention protocols",
            parameters=[
                ToolParameter(
                    name="disease_name",
                    description="Suspected pathogen or physiological condition",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="treatment",
                    description="Recommended remedial treatment (e.g. Copper Spray, Bio-fungicide)",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="urgency",
                    description="Alert priority level: LOW, MEDIUM, HIGH, CRITICAL",
                    param_type="str",
                    required=False,
                    default="MEDIUM",
                ),
                ToolParameter(
                    name="zone",
                    description="Target farm or greenhouse zone identifier",
                    param_type="str",
                    required=True,
                ),
                ToolParameter(
                    name="notes",
                    description="Additional application instructions or PPE requirements",
                    param_type="str",
                    required=False,
                    default="",
                ),
            ],
            returns="Structured alert dispatch receipt with unique alert ID and timestamp",
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Dispatch treatment alert to farm management subsystem.

        Args:
            **kwargs: Must contain ``disease_name``, ``treatment``, and ``zone``.

        Returns:
            ``ToolResult`` with dispatch receipt.
        """
        disease_name = str(kwargs.get("disease_name", "")).strip()
        treatment = str(kwargs.get("treatment", "")).strip()
        urgency = str(kwargs.get("urgency", "MEDIUM")).upper().strip()
        zone = str(kwargs.get("zone", "")).strip()
        notes = str(kwargs.get("notes", ""))

        if not disease_name:
            return ToolResult(success=False, error="Missing required parameter 'disease_name'")
        if not treatment:
            return ToolResult(success=False, error="Missing required parameter 'treatment'")
        if not zone:
            return ToolResult(success=False, error="Missing required parameter 'zone'")
        if urgency not in self.ALLOWED_URGENCIES:
            urgency = "MEDIUM"

        alert_id = f"treat_alert_{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()

        logger.info(
            "TreatmentAlertTool dispatched alert [%s]: %s -> %s (urgency=%s, zone=%s)",
            alert_id,
            disease_name,
            treatment,
            urgency,
            zone,
        )

        return ToolResult(
            success=True,
            data={
                "tool": "treatment_alert",
                "alert_id": alert_id,
                "disease_name": disease_name,
                "treatment": treatment,
                "urgency": urgency,
                "zone": zone,
                "notes": notes,
                "status": "DISPATCHED",
                "timestamp": ts,
            },
            metadata={"requires_human_verification": urgency in ("HIGH", "CRITICAL")},
        )
