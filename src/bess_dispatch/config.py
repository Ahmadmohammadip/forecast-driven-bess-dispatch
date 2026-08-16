"""Configuration: experiments are described in YAML, not in argument lists.

The brief asks for config-driven experiments and one command that reproduces
the baseline. A scenario is then a small YAML file that overrides part of
`configs/base.yaml`, which keeps the diff between two experiments readable —
you can see what changed without reading any Python.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bess_dispatch.data.schema import Battery, GridConnection, SiteConfig, TariffPolicy

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge, so a scenario can override one nested key."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully resolved experiment."""

    name: str
    battery: dict[str, Any]
    grid: dict[str, Any]
    tariff: dict[str, Any]
    objective: str = "cost_degradation_demand"
    horizon: int = 24
    dt_hours: float = 1.0
    enforce_terminal_soc: bool = True
    split: str = "test"
    models: dict[str, str] = field(default_factory=dict)
    seed: int = 20260816
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def site(self) -> SiteConfig:
        """Build the validated `SiteConfig` this experiment describes."""
        return SiteConfig(
            battery=Battery(**self.battery),
            grid=GridConnection(**self.grid),
            tariff_policy=TariffPolicy(**self.tariff),
            dt_hours=self.dt_hours,
            enforce_terminal_soc=self.enforce_terminal_soc,
        )

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        """A copy with nested overrides applied, for sensitivity sweeps."""
        merged = _deep_merge(self.raw, overrides)
        return _from_mapping(merged)


def _from_mapping(mapping: dict[str, Any]) -> ExperimentConfig:
    required = ("battery", "grid", "tariff")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise KeyError(f"config is missing required section(s): {missing}")
    return ExperimentConfig(
        name=mapping.get("name", "unnamed"),
        battery=mapping["battery"],
        grid=mapping["grid"],
        tariff=mapping["tariff"],
        objective=mapping.get("objective", "cost_degradation_demand"),
        horizon=mapping.get("horizon", 24),
        dt_hours=mapping.get("dt_hours", 1.0),
        enforce_terminal_soc=mapping.get("enforce_terminal_soc", True),
        split=mapping.get("split", "test"),
        models=mapping.get("models", {}),
        seed=mapping.get("seed", 20260816),
        raw=mapping,
    )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a config, merging it over `configs/base.yaml` if it names a parent.

    A scenario file states `extends: base.yaml` and then only the keys it
    changes, so the file itself documents the difference from the baseline.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}

    parent = mapping.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        if not parent_path.exists():
            parent_path = CONFIG_DIR / parent
        with parent_path.open(encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        base.pop("extends", None)
        mapping = _deep_merge(base, mapping)

    return _from_mapping(mapping)


def available_scenarios() -> list[Path]:
    scenario_dir = CONFIG_DIR / "scenarios"
    return sorted(scenario_dir.glob("*.yaml")) if scenario_dir.exists() else []
