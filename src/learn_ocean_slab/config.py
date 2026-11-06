"""Configuration loading helpers."""

import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .physics import coriolis_parameter

SECONDS_PER_DAY = 86400.0

# better path resolution
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MOVIE_DIR = Path(__file__).resolve().parents[2] / "movies"


def load_config(path: str | Path, overrides: list[str] = ()) -> "RunConfig":
    """Load a YAML configuration file, then apply overrides like "lat=30" or "u_initial=0.5".

    Every shared RunConfig field must be in the file; model-specific fields may be left out.
    A misspelled name, a missing shared field or a value of the wrong type raises an error.
    """
    data = OmegaConf.load(Path(path))
    if not isinstance(data, DictConfig):
        raise ValueError("YAML config must contain a top-level mapping")
    schema = OmegaConf.structured(RunConfig)  # field names and types, no values
    merged = OmegaConf.merge(schema, data, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)


def check_model_fields(config, model, needs=()):
    """Raise an error if a model-specific field the model needs is missing; warn about ones it ignores.

    model: the model's name, for the messages. needs: names of the model-specific fields it uses.
    """
    missing = [name for name in needs if getattr(config, name) is None]
    if missing:
        raise ValueError(f"{model} needs {', '.join(missing)} in the config")
    model_specific = [field.name for field in fields(config) if field.default is None]
    ignored = [name for name in model_specific if name not in needs and getattr(config, name) is not None]
    if ignored:
        warnings.warn(f"{model} ignores {', '.join(ignored)}", stacklevel=3)  # points at the caller of run()


@dataclass(frozen=True)
class RunConfig:
    """Overarching slab model data class"""

    model: str  # which slab model runs this config, e.g. pollard_millard
    lat: float  # latitude [degrees N]
    H: float  # mixed layer depth [m]; the initial depth in models where it evolves
    rho: float  # density [kg/m^3]
    tau_x: float  # wind stress (east) [N/m^2]
    tau_y: float  # wind stress (north)[N/m^2]
    zeta_over_f: float  # background vorticity over f, zeta/f; front runs east-west, jet U(y), zeta = -dU/dy
    damping_days: float  # pollard-millard friction term [days]
    duration_days: float  # length of the run [days]
    dt: float  # time step [s]
    step_type: str  # "exact" (step_exact) or "rk4" (step_rk4)
    forcing: str  # "average" (step_exact's treatment) or "linear"; how wind is treated within a step
    u_initial: float  # initial east velocity [m/s]
    v_initial: float  # initial north velocity [m/s]
    wind_type: str  # steady, rotating or oscillating
    wind_period_hours: float  # period of rotation/oscillation [hours]; negative = counterclockwise
    wind_duration_hours: float  # how long the wind blows [hours]; .inf = never stops
    gust_std: float  # random gust size, per component [N/m^2]; 0 = no gusts
    gust_hours: float  # typical gust duration [hours]
    seed: int  # random seed for the gusts

    # Model-specific fields: not every model uses them, so a config may leave them out (None means "not
    # given", not a default value). Each model checks the ones it needs with check_model_fields.
    N2: float | None = None  # stratification below the mixed layer, N^2 [1/s^2]
    Ri_critical: float | None = None  # bulk Richardson number below which the mixed layer deepens

    def __repr__(self) -> str:
        lines = [f"{field.name} = {getattr(self, field.name)}" for field in fields(self)]
        lines += [f"f = {self.f:.4e} 1/s", f"r = {self.r:.4e} 1/s"]
        return "RunConfig(\n  " + "\n  ".join(lines) + "\n)"

    @property
    def r(self) -> float:
        return 1.0 / (self.damping_days * SECONDS_PER_DAY)

    @property
    def duration(self) -> float:
        return self.duration_days * SECONDS_PER_DAY

    @property
    def wind_omega(self) -> float:
        """Wind rotation/oscillation frequency 2 pi / period [rad/s]; 0 for an infinite period."""
        return 2 * np.pi / (self.wind_period_hours * 3600)

    @property
    def wind_duration(self) -> float:
        """How long the wind blows [s]."""
        return self.wind_duration_hours * 3600

    @property
    def inertial_period(self) -> float:
        """Inertial period 2 pi / |f| [s]."""
        return 2 * np.pi / abs(self.f)

    @property
    def f(self) -> float:
        """Coriolis parameter at this run's latitude [1/s]."""
        return coriolis_parameter(self.lat)

    @property
    def zeta(self) -> float:
        """Background relative vorticity [1/s]."""
        return self.zeta_over_f * self.f

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of the inputs that were given, for saving alongside a run's output."""
        return {name: value for name, value in asdict(self).items() if value is not None}
