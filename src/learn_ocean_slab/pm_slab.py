"""
Pollard-Millard Slab model (based on their 1970 paper)
"""

import numpy as np
import xarray as xr

from .forcing import wind_from_config


def exp_ix(x):
    return np.exp(1j * x)
    # return np.cos(x)+ np.sin(x)*(0+1j) # the same thing


def pm_model_current_response_constant_wind(t, tau_over_rho_h, f, r, s_0):
    """

    The Pollard-Millard model is given by
    du/dt - f v = tau x/(rho H) - r u
    dv/dt + f u = tau y/(rho H) - r v
    where r is a 'damping term' that opposes the current motion
    and tau is a wind forcing.

    - this can be combined to:
    d(u + iv)/dt  +  (-f v + i f u)  =  (Tx + i Ty)  - r (u + iv)
    d(u + iv)/dt  +  if(i v + u)  =  (Tx + i Ty)  - r (u + iv)

    - substitute s = u+iv and rearrange
    ds/dt + (if+r)s = (Tx+i Ty) = T

    - The homogenous solution
     s_g(t) = A exp(-(if+r)*t)

    - Try a particular solution s_p = C, then ds_p/dt =0
    (if+r) s_p = T , or s_p = T/(if+r)

    - solution:
    s(t) = s_g(t) + s_p(t) = A exp (-(if+r)*t)+ T/(if+r)

    - A is given by the initial condition s(t=0)
    s(t) = (s_0- T/(if+r)) exp(-(if+r)*t) + T/(if+r)

    - we can rewrite:
    s(t) = s_0 *exp(-if*t)*exp(-rt) + T/(if+r) * (1- exp(-if*t)*exp(-rt))




    t = times[s] (array),
    tau_over_rho_h: forcing (tau_x + i tau_y)/rho/H
    f: coriolis [1/s]
    r: damping rate [1/s]
    s_0: initial velocity

    returns: s = u + iv [m/s]

    """
    return s_0 * (exp_ix(-f * t)) * np.exp(-r * t) + tau_over_rho_h / (f * (1j) + r) * (
        1 - exp_ix(-f * t) * np.exp(-r * t)
    )


def pm_model_instant_transport_constant_wind(s, H):
    """depth integrated transport [m^2/s] - since nothing depends on Z,  int( s(t) dz) = H s(t)"""
    return H * s


def pm_model_integrated_transport_constant_wind(t, tau_over_rho_h, f, r, s_0, H):
    """Transport accumulated since t=0, int_0^t M dt' [m^2], i.e. m^3 of water per m of width.

    - From the current solution, with s_p = T/(if+r):
    s(t') = s_p + (s_0 - s_p) exp(-(if+r)*t')

    - Integrate each term, using int_0^t exp(-(if+r)*t') dt' = (1 - exp(-(if+r)*t))/(if+r):
    int_0^t s dt' = s_p*t + (s_0 - s_p) * (1 - exp(-(if+r)*t))/(if+r)

    - Multiply by H to go from velocity to transport.
    Divided by H, this is the displacement of a water parcel in the slab [m].
    """
    s_p = tau_over_rho_h / (1j * f + r)
    return H * (s_p * t + (s_0 - s_p) * (1 - np.exp(-(1j * f + r) * t)) / (1j * f + r))


def pm_model_integrated_speed_squared_constant_wind(t, tau_over_rho_h, f, r, s_0):
    """int_0^t |s|^2 dt' for a constant wind [m^2/s]; times rho H r it is the energy removed by damping.

    - From the current solution, with s_p = T/(if+r) and D = s_0 - s_p:
    s(t') = s_p + D exp(-(if+r)*t')

    - Expand |s|^2 = s times its complex conjugate:
    |s|^2 = |s_p|^2 + 2 Re(conj(s_p) D exp(-(if+r)*t')) + |D|^2 exp(-2r*t')
    The last term has no f in it: the rotation exp(-if*t') has size 1, so Coriolis does no work.

    - Integrate each term:
    int_0^t |s|^2 dt' = |s_p|^2 t + 2 Re(conj(s_p) D (1 - exp(-(if+r)*t))/(if+r)) + |D|^2 (1 - exp(-2r*t))/(2r)
    With r = 0 the last factor (1 - exp(-2r*t))/(2r) becomes t.
    """
    s_p = tau_over_rho_h / (1j * f + r)
    D = s_0 - s_p
    decay = t if r == 0 else (1 - np.exp(-2 * r * t)) / (2 * r)
    cross = (s_p.conjugate() * D * (1 - np.exp(-(1j * f + r) * t)) / (1j * f + r)).real
    return abs(s_p) ** 2 * t + 2 * cross + abs(D) ** 2 * decay


def step_exact(dt, s_n, tau_mid, rho, H, f, r):
    """Advance the slab over one interval of length dt with the wind held at tau_mid, exactly.

    Returns, for this interval: the current at its end [m/s], and how much parcel displacement [m],
    wind energy input [J/m^2] and damping energy loss [J/m^2] build up over it.
    """
    T = tau_mid / (rho * H)
    s_next = pm_model_current_response_constant_wind(dt, T, f, r, s_n)
    displacement = pm_model_integrated_transport_constant_wind(dt, T, f, r, s_n, H) / H  # int s dt'
    # tau is constant over the interval, so int tau . u dt' = Re(conj(tau) int s dt')
    wind_input = (tau_mid.conjugate() * displacement).real
    damping_loss = rho * H * r * pm_model_integrated_speed_squared_constant_wind(dt, T, f, r, s_n)
    return s_next, displacement, wind_input, damping_loss


def integrate(t, tau, rho, H, f, r, s_0, step=step_exact):
    """Run the slab through a wind stress time series, one interval at a time.

    Over each interval the wind is held at its interval average, and `step` advances the current,
    parcel position, wind energy input and damping energy loss together, so all of them are
    integrated the same way.
    t: times [s], tau: wind stress tau_x + i tau_y at each time [N/m^2], s_0: velocity at t[0] [m/s].
    Returns s [m/s], position x + iy [m], and wind input and damping loss since t[0] [J/m^2].
    """
    tau_mid = (tau[:-1] + tau[1:]) / 2  # interval-averaged wind
    s = np.zeros(len(t), dtype=complex)
    position = np.zeros(len(t), dtype=complex)
    wind_input = np.zeros(len(t))
    damping_loss = np.zeros(len(t))
    s[0] = s_0
    for n, dt in enumerate(np.diff(t)):
        s[n + 1], d_position, d_wind_input, d_damping_loss = step(dt, s[n], tau_mid[n], rho, H, f, r)
        position[n + 1] = position[n] + d_position
        wind_input[n + 1] = wind_input[n] + d_wind_input
        damping_loss[n + 1] = damping_loss[n] + d_damping_loss
    return s, position, wind_input, damping_loss


def run(config):
    """Run the slab model for one configuration. Returns an xarray Dataset."""
    t = np.arange(0, config.duration, config.dt)  # [s]
    tau = wind_from_config(t, config)  # wind stress [N/m^2]
    s_0 = config.u_initial + 1j * config.v_initial  # [m/s]
    s, position, wind_energy_input, damping_energy_loss = integrate(
        t, tau, config.rho, config.H, config.f, config.r, s_0
    )  # position: water parcel displacement [m]

    T = tau / (config.rho * config.H)  # forcing [m/s^2]
    s_steady = T / (1j * config.f + config.r)  # steady state for the wind at each time [m/s]
    M = config.H * s  # depth-integrated transport [m^2/s]

    wind_work = (tau.conjugate() * s).real  # tau_x u + tau_y v, energy input from the wind [W/m^2]
    kinetic_energy = 0.5 * config.rho * config.H * np.abs(s) ** 2  # slab kinetic energy per unit area [J/m^2]
    damping_loss = config.rho * config.H * config.r * np.abs(s) ** 2  # energy removed by the damping [W/m^2]
    angle_to_wind = np.where(
        tau != 0, np.degrees(np.angle(s * tau.conjugate())), np.nan
    )  # current direction minus wind direction [degrees]; NaN with no wind

    return xr.Dataset(
        {
            "u": ("time", s.real, {"units": "m/s", "long_name": "eastward velocity"}),
            "v": ("time", s.imag, {"units": "m/s", "long_name": "northward velocity"}),
            "tau_x": ("time", tau.real, {"units": "N/m^2", "long_name": "eastward wind stress"}),
            "tau_y": ("time", tau.imag, {"units": "N/m^2", "long_name": "northward wind stress"}),
            "u_steady": ("time", s_steady.real, {"units": "m/s", "long_name": "eastward steady-state velocity"}),
            "v_steady": ("time", s_steady.imag, {"units": "m/s", "long_name": "northward steady-state velocity"}),
            "M_x": ("time", M.real, {"units": "m^2/s", "long_name": "eastward transport"}),
            "M_y": ("time", M.imag, {"units": "m^2/s", "long_name": "northward transport"}),
            "wind_work": ("time", wind_work, {"units": "W/m^2", "long_name": "wind work on the slab"}),
            "kinetic_energy": ("time", kinetic_energy, {"units": "J/m^2", "long_name": "slab kinetic energy"}),
            "damping_loss": ("time", damping_loss, {"units": "W/m^2", "long_name": "energy removed by damping"}),
            "wind_energy_input": (
                "time",
                wind_energy_input,
                {"units": "J/m^2", "long_name": "total energy put in by the wind since the start"},
            ),
            "damping_energy_loss": (
                "time",
                damping_energy_loss,
                {"units": "J/m^2", "long_name": "total energy removed by damping since the start"},
            ),
            "angle_to_wind": (
                "time",
                angle_to_wind,
                {
                    "units": "degrees",
                    "long_name": "current direction relative to the wind, positive = left of the wind",
                },
            ),
            "x": ("time", position.real, {"units": "m", "long_name": "eastward parcel displacement"}),
            "y": ("time", position.imag, {"units": "m", "long_name": "northward parcel displacement"}),
        },
        coords={"time": ("time", t, {"units": "s"})},
        attrs=config.to_dict(),
    )
