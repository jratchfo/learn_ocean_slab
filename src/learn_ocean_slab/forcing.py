"""Wind stress time series for driving the slab model.

Every function returns complex wind stress tau_x + i tau_y [N/m^2] at each time in t [s].
They combine: a storm that turns is wind_pulse(...) * np.exp(-1j * omega * t).
"""

import numpy as np


def wind_pulse(t, tau, duration):
    """Wind that blows at tau [N/m^2] from t = 0 until `duration` [s], then stops.

    duration = np.inf gives a wind that never stops.
    """
    return np.where(t < duration, tau, 0)


def wind_rotating(t, tau, omega):
    """Constant-strength wind whose direction turns at rate omega [rad/s].

    tau(t) = tau exp(-i omega t): clockwise for omega > 0 (the Northern Hemisphere
    inertial direction), counterclockwise for omega < 0.
    """
    return tau * np.exp(-1j * omega * t)


def wind_oscillating(t, tau, omega):
    """Wind that swings back and forth along the direction of tau at frequency omega [rad/s].

    tau cos(omega t) = tau/2 (e^{-i omega t} + e^{i omega t}): half clockwise, half counterclockwise.
    """
    return tau * np.cos(omega * t)


def wind_gusts(t, gust_std, gust_time, seed):
    """Random gust fluctuations, to be added to a mean wind.

    Each component is a red-noise (Ornstein-Uhlenbeck) process: standard deviation gust_std [N/m^2],
    and gusts last about gust_time [s]. The same seed always gives the same gusts.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))
    gusts = np.zeros(len(t), dtype=complex)
    gusts[0] = gust_std * noise[0]
    for n, dt in enumerate(np.diff(t)):
        memory = np.exp(-dt / gust_time)  # how much of the previous gust survives this step
        gusts[n + 1] = memory * gusts[n] + gust_std * np.sqrt(1 - memory**2) * noise[n + 1]
    return gusts


def wind_from_config(t, config):
    """Wind stress time series described by a RunConfig's wind settings."""
    tau = config.tau_x + 1j * config.tau_y
    if config.wind_type == "steady":
        wind = np.full(len(t), tau)
    elif config.wind_type == "rotating":
        wind = wind_rotating(t, tau, config.wind_omega)
    elif config.wind_type == "oscillating":
        wind = wind_oscillating(t, tau, config.wind_omega)
    else:
        raise ValueError(f"unknown wind_type {config.wind_type!r}: use steady, rotating or oscillating")

    if config.gust_std > 0:
        wind = wind + wind_gusts(t, config.gust_std, config.gust_hours * 3600, config.seed)
    return wind_pulse(t, wind, config.wind_duration)  # gusts stop when the storm does
