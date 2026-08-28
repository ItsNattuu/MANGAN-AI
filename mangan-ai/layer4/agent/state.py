from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    request: str = ""

    layer2_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    layer3_results: list[dict[str, Any]] = field(
        default_factory=list
    )

    validated_targets: list[dict[str, Any]] = field(
        default_factory=list
    )

    best_target: dict[str, Any] | None = None

    decision: str = ""

    recommendation: str = ""

    reasoning: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )
