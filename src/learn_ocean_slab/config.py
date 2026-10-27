"""Configuration loading helpers."""
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .physics import coriolis_parameter

SECONDS_PER_DAY=86400.0

# better path resolution
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MOVIE_DIR = Path(__file__).resolve().parents[2] / "movies"

def load_config(path: str | Path, overrides: list[str] = (), config_class: type = None) -> "RunConfig":
    """Load a YAML configuration file, then apply overrides like "lat=30" or "u_initial=0.5".

    config_class is the model's config dataclass (RunConfig if not given). Every one of its
    fields must be in the file; overrides only change values that are there.
    A misspelled name, a missing field or a value of the wrong type raises an error.
    """
    data = OmegaConf.load(Path(path))
    if not isinstance(data, DictConfig):
        raise ValueError("YAML config must contain a top-level mapping")
    schema = OmegaConf.structured(config_class or RunConfig)  # field names and types, no values
    merged = OmegaConf.merge(schema, data, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)

@dataclass(frozen=True)
class RunConfig:
    """Overarching slab model data class"""
    model: str             # which slab model runs this config, e.g. pollard_millard
    lat: float             # latitude [degrees N]
    H: float               # mixed layer depth [m]
    rho: float             # density [kg/m^3]
    tau_x: float           # wind stress (east) [N/m^2]
    tau_y: float           # wind stress (north)[N/m^2]
    damping_days: float    # pollard-millard friction term [days]
    duration_days: float   # length of the run [days]
    dt: float              # time step [s]
    u_initial: float       # initial east velocity [m/s]
    v_initial: float       # initial north velocity [m/s]
    wind_type: str         # steady, rotating or oscillating
    wind_period_hours: float    # period of rotation/oscillation [hours]; negative = counterclockwise
    wind_duration_hours: float  # how long the wind blows [hours]; .inf = never stops
    gust_std: float        # random gust size, per component [N/m^2]; 0 = no gusts
    gust_hours: float      # typical gust duration [hours]
    seed: int              # random seed for the gusts

    def __repr__(self) -> str:
        lines = [f"{field.name} = {getattr(self, field.name)}" for field in fields(self)]
        lines += [f"f = {self.f:.4e} 1/s", f"r = {self.r:.4e} 1/s"]
        return "RunConfig(\n  " + "\n  ".join(lines) + "\n)"
    @property
    def r(self)-> float:
        return 1.0/(self.damping_days * SECONDS_PER_DAY)

    @property
    def duration(self)-> float:
        return (self.duration_days * SECONDS_PER_DAY)
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

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of the inputs, for saving alongside a run's output."""
        return asdict(self)
