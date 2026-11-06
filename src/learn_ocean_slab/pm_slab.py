"""
Pollard-Millard Slab Model

R.T. Pollard, R.C. Millard, Comparison between observed and simulated wind-generated
inertial oscillations, Deep Sea Research and Oceanographic Abstracts,
Volume 17, Issue 4, 1970, Pages 813-821, https://doi.org/10.1016/0011-7471(70)90043-4.

"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import xarray as xr

from .config import check_model_fields
from .forcing import wind_from_config
from .integrators import rk4_step


@dataclass
class SlabState:
    """Bundled state for step_rk4: the stretched current plus every quantity integrated alongside
    it, so rk4_step advances all of them with the same accuracy. s_hat is in the stretched frame
    (see step_exact); the other four are already physical [SI], not stretched.

    rk4_step only ever calls + and * on whatever state object it's given, so defining those two
    here is all it takes for rk4_step to work on this bundle exactly as it would on a bare number.
    """

    s_hat: complex
    displacement: complex
    wind_input: float
    damping_loss: float
    shear_exchange: float

    def __add__(self, other):
        return SlabState(*(a + b for a, b in zip(vars(self).values(), vars(other).values())))

    def __mul__(self, scalar):
        return SlabState(*(scalar * v for v in vars(self).values()))

    __rmul__ = __mul__


def stretch_forcing(tau, rho, H, gamma):
    """tau/(rho*H), with its imaginary part (the v-component's forcing) stretched by gamma.

    Shared by step_exact (for the forcing at the interval's endpoints) and wind_at (for the
    forcing at an arbitrary time within it) - see front_stretch/step_exact for what gamma is.
    """
    T = tau / (rho * H)
    return T.real + 1j * gamma * T.imag


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

def pm_model_current_response_linear_wind(t, tau_over_rho_h1,tau_over_rho_h2, f, r, s_0):
    """[add description of solution here]
    """
    k = r + 1j*f
    a = tau_over_rho_h1
    b = (tau_over_rho_h2 - tau_over_rho_h1) / t

    return (s_0 - a/k + b/k**2) * np.exp(-k*t) + a/k - b/k**2 + (b/k)*t

def constant_wind_solution(tau_over_rho_h, f, r, s_0):
    """Pieces of the constant-wind solution s(t') = s_p + D exp(-k*t'), shared by the integrals below.

    Returns k = if + r [1/s], the steady state s_p = T/k [m/s] and the starting departure from it,
    D = s_0 - s_p [m/s].
    """
    k = 1j * f + r
    s_p = tau_over_rho_h / k
    return k, s_p, s_0 - s_p


def integrated_exp(k, t):
    """int_0^t exp(-k*t') dt' = (1 - exp(-k*t))/k, which becomes t when k = 0."""
    return t if k == 0 else (1 - np.exp(-k * t)) / k


def integrated_t_exp(k, t):
    """int_0^t t' exp(-k*t') dt' = (1 - exp(-k*t)*(1+k*t))/k^2, which becomes t^2/2 when k = 0."""
    return t**2 / 2 if k == 0 else (1 - np.exp(-k * t) * (1 + k * t)) / k**2


def linear_wind_solution(tau_over_rho_h1, tau_over_rho_h2, t, f, r, s_0):
    """Pieces of the linear-forcing solution s(t') = C exp(-k*t') + P0 + P1*t', shared by the
    integrals below. tau_over_rho_h1, tau_over_rho_h2: forcing at the start and end of the interval
    of length t, interpolated linearly between them.

    Returns k = if + r [1/s], and the exponential's coefficient C and the particular solution's
    pieces P0, P1 [m/s] (the particular solution is linear in time since the forcing is).
    """
    k = 1j * f + r
    a = tau_over_rho_h1
    b = (tau_over_rho_h2 - tau_over_rho_h1) / t
    P0 = a / k - b / k**2
    P1 = b / k
    return k, s_0 - P0, P0, P1


def pm_model_integrated_transport_linear_wind(t, tau_over_rho_h1, tau_over_rho_h2, f, r, s_0, H):
    """int_0^t H*s dt' [m^2] for linearly-interpolated forcing; divided by H this is the parcel
    displacement [m]. Same derivation as pm_model_integrated_transport_constant_wind, but integrating
    the extra P1*t' term of the linear-forcing solution too.
    """
    k, C, P0, P1 = linear_wind_solution(tau_over_rho_h1, tau_over_rho_h2, t, f, r, s_0)
    return H * (C * integrated_exp(k, t) + P0 * t + P1 * t**2 / 2)


def pm_model_integrated_moment_linear_wind(t, tau_over_rho_h1, tau_over_rho_h2, f, r, s_0):
    """int_0^t t'*s dt' [m] for linearly-interpolated forcing; the first moment in time of s, needed
    because the wind itself varies over the interval (wind_input can no longer pull a constant tau
    outside the integral, see step_exact).
    """
    k, C, P0, P1 = linear_wind_solution(tau_over_rho_h1, tau_over_rho_h2, t, f, r, s_0)
    return C * integrated_t_exp(k, t) + P0 * t**2 / 2 + P1 * t**3 / 3


def pm_model_integrated_speed_squared_linear_wind(t, tau_over_rho_h1, tau_over_rho_h2, f, r, s_0):
    """int_0^t |s|^2 dt' for linearly-interpolated forcing [m^2/s].

    |s|^2 = |C|^2 exp(-2r*t') + 2 Re(C conj(P0) exp(-k*t')) + 2 Re(C conj(P1) t' exp(-k*t'))
            + |P0|^2 + 2 Re(P0 conj(P1)) t' + |P1|^2 t'^2
    using k + conj(k) = 2r (the rotation exp(-i*omega*t') has size 1, so only r sets the |C|^2 decay).
    """
    k, C, P0, P1 = linear_wind_solution(tau_over_rho_h1, tau_over_rho_h2, t, f, r, s_0)
    return (
        abs(C) ** 2 * integrated_exp(2 * r, t)
        + 2 * (C * P0.conjugate() * integrated_exp(k, t)).real
        + 2 * (C * P1.conjugate() * integrated_t_exp(k, t)).real
        + abs(P0) ** 2 * t
        + (P0 * P1.conjugate()).real * t**2
        + abs(P1) ** 2 * t**3 / 3
    )


def pm_model_integrated_square_linear_wind(t, tau_over_rho_h1, tau_over_rho_h2, f, r, s_0):
    """int_0^t s^2 dt' for linearly-interpolated forcing [m^2/s]; s squared as a complex number.

    s^2 = C^2 exp(-2k*t') + 2 C P0 exp(-k*t') + 2 C P1 t' exp(-k*t') + P0^2 + 2 P0 P1 t' + P1^2 t'^2
    """
    k, C, P0, P1 = linear_wind_solution(tau_over_rho_h1, tau_over_rho_h2, t, f, r, s_0)
    return (
        C**2 * integrated_exp(2 * k, t)
        + 2 * C * P0 * integrated_exp(k, t)
        + 2 * C * P1 * integrated_t_exp(k, t)
        + P0**2 * t
        + P0 * P1 * t**2
        + P1**2 * t**3 / 3
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
    k, s_p, D = constant_wind_solution(tau_over_rho_h, f, r, s_0)
    return H * (s_p * t + D * integrated_exp(k, t))


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
    k, s_p, D = constant_wind_solution(tau_over_rho_h, f, r, s_0)
    cross = (s_p.conjugate() * D * integrated_exp(k, t)).real
    return abs(s_p) ** 2 * t + 2 * cross + abs(D) ** 2 * integrated_exp(2 * r, t)


def pm_model_integrated_square_constant_wind(t, tau_over_rho_h, f, r, s_0):
    """int_0^t s^2 dt' for a constant wind [m^2/s]; s squared as a complex number, not |s|^2.

    Needed once the oscillation is an ellipse: Re(s^2) = u^2 - v^2 and Im(s^2) = 2 u v.

    - From the current solution, with s_p = T/(if+r), D = s_0 - s_p and k = if + r:
    s(t') = s_p + D exp(-k*t')

    - Square it:
    s^2 = s_p^2 + 2 s_p D exp(-k*t') + D^2 exp(-2k*t')

    - Integrate each term:
    int_0^t s^2 dt' = s_p^2 t + 2 s_p D (1 - exp(-k*t))/k + D^2 (1 - exp(-2k*t))/(2k)
    Same as int |s|^2 but without the conjugate, so the last term decays at 2k instead of 2r.
    """
    k, s_p, D = constant_wind_solution(tau_over_rho_h, f, r, s_0)
    return s_p**2 * t + 2 * s_p * D * integrated_exp(k, t) + D**2 * integrated_exp(2 * k, t)



def front_stretch(f, zeta):
    """Stretch factor gamma = sqrt((f + zeta)/f) and inertial frequency omega = f gamma [1/s] in vorticity zeta.

    omega = sqrt(f (f + zeta)) with the sign of f; see step_exact for how they are used.
    """
    if 1 + zeta / f <= 0:
        raise ValueError(f"zeta/f = {zeta / f:.2f} <= -1: the front is inertially unstable")
    gamma = np.sqrt(1 + zeta / f)
    return gamma, f * gamma


def step_exact(dt, s_n, tau_n, tau_np1, rho, H, f, r, zeta, forcing="average" ):
    """Advance the slab over one interval of length dt with the wind held at its interval average, exactly.

    With a background vorticity zeta (front running east-west, jet U(y), zeta = -dU/dy) the slab obeys
    du/dt - (f + zeta) v = Tx - r u
    dv/dt +  f       u = Ty - r v

    - Stretch v by gamma = sqrt((f + zeta)/f): the stretched current s_hat = u + i gamma v obeys the
    Pollard-Millard equation
    d(s_hat)/dt + (r + i omega) s_hat = Tx + i gamma Ty,   with omega = f gamma = sqrt(f (f + zeta)) (sign of f)
    so the Pollard-Millard solutions give s_hat, and u = Re s_hat, v = Im s_hat / gamma.
    With zeta = 0, gamma = 1 and s_hat = s: plain Pollard-Millard.

    - The energy integrals need u^2 + v^2 and u v; in terms of s_hat,
    u^2 + v^2 = (1/2)(1 + 1/gamma^2)|s_hat|^2 + (1/2)(1 - 1/gamma^2) Re(s_hat^2),   u v = Im(s_hat^2) / (2 gamma)

    Returns, for this interval: the current at its end [m/s], and how much parcel displacement [m],
    wind energy input [J/m^2], damping energy loss [J/m^2] and energy taken from the front's jet
    (rho H zeta int u v dt') [J/m^2] build up over it.
    """
    gamma, omega = front_stretch(f, zeta)  # omega is f_eff [1/s]
    T_hat_n = stretch_forcing(tau_n, rho, H, gamma)
    T_hat_np1 = stretch_forcing(tau_np1, rho, H, gamma)

    s_hat_n = s_n.real + 1j * gamma * s_n.imag

    if forcing=="average":
        T_hat = (T_hat_n + T_hat_np1) / 2
        s_hat_next = pm_model_current_response_constant_wind(dt, T_hat, omega, r, s_hat_n)

        
        integrated_s_hat = pm_model_integrated_transport_constant_wind(dt, T_hat, omega, r, s_hat_n, H) / H
        displacement = integrated_s_hat.real + 1j * integrated_s_hat.imag / gamma  # int s dt'
        
        # tau is constant over the interval, so int tau . u dt' = Re(conj(tau) int s dt')
        wind_input = (((tau_n + tau_np1) / 2).conjugate() * displacement).real

        # int |s_hat|^2 dt' and int s_hat^2 dt'
        integrated_s_hat_abs2 = pm_model_integrated_speed_squared_constant_wind(dt, T_hat, omega, r, s_hat_n)
        integrated_s_hat2 = pm_model_integrated_square_constant_wind(dt, T_hat, omega, r, s_hat_n)
        integrated_speed2 = (
            0.5 * (1 + 1 / gamma**2) * integrated_s_hat_abs2 + 0.5 * (1 - 1 / gamma**2) * integrated_s_hat2.real
        )
        damping_loss = rho * H * r * integrated_speed2
        shear_exchange = rho * H * zeta * integrated_s_hat2.imag / (2 * gamma)
    
    elif forcing=="linear":
        s_hat_next = pm_model_current_response_linear_wind(dt, T_hat_n, T_hat_np1, omega, r, s_hat_n)

        integrated_s_hat = pm_model_integrated_transport_linear_wind(dt, T_hat_n, T_hat_np1, omega, r, s_hat_n, H) / H
        displacement = integrated_s_hat.real + 1j * integrated_s_hat.imag / gamma  # int s dt'

        # tau varies over the interval now, so wind_input needs the time-weighted moment too, not just displacement
        moment_s_hat = pm_model_integrated_moment_linear_wind(dt, T_hat_n, T_hat_np1, omega, r, s_hat_n)
        p, q = tau_n, (tau_np1 - tau_n) / dt
        I1 = p.conjugate() * integrated_s_hat + q.conjugate() * moment_s_hat
        I2 = (p * integrated_s_hat + q * moment_s_hat).conjugate()
        wind_input = ((gamma + 1) / (2 * gamma) * I1 + (gamma - 1) / (2 * gamma) * I2).real

        integrated_s_hat_abs2 = pm_model_integrated_speed_squared_linear_wind(dt, T_hat_n, T_hat_np1, omega, r, s_hat_n)
        integrated_s_hat2 = pm_model_integrated_square_linear_wind(dt, T_hat_n, T_hat_np1, omega, r, s_hat_n)
        integrated_speed2 = (
            0.5 * (1 + 1 / gamma**2) * integrated_s_hat_abs2 + 0.5 * (1 - 1 / gamma**2) * integrated_s_hat2.real
        )
        damping_loss = rho * H * r * integrated_speed2
        shear_exchange = rho * H * zeta * integrated_s_hat2.imag / (2 * gamma)
    s_next = s_hat_next.real + 1j * s_hat_next.imag / gamma

    return s_next, displacement, wind_input, damping_loss, shear_exchange

def wind_at(t, tau_n, tau_np1, dt, rho, H, gamma, forcing):
    """Physical wind stress tau(t) and its stretched, rho*H-scaled form T_hat(t) at time t in [0, dt].

    forcing == "average" holds tau at the interval's average (step_exact's treatment, for comparison);
    anything else interpolates it linearly between tau_n (t=0) and tau_np1 (t=dt).
    """
    tau = (tau_n + tau_np1) / 2 if forcing == "average" else tau_n + (tau_np1 - tau_n) * t / dt
    return tau, stretch_forcing(tau, rho, H, gamma)


def slab_rhs(t, state, tau_n, tau_np1, dt, rho, H, r, zeta, gamma, k, forcing):
    """d(state)/dt for step_rk4: the Pollard-Millard equation for s_hat, plus the instantaneous rate
    of each energy/position quantity bundled alongside it in state (see SlabState).

    t, state are the two arguments rk4_step always calls this with; every other parameter step_rk4
    needs is bound in ahead of time with functools.partial, so this stays an ordinary top-level
    function instead of a closure.
    """
    tau, T_hat = wind_at(t, tau_n, tau_np1, dt, rho, H, gamma, forcing)
    s = state.s_hat.real + 1j * state.s_hat.imag / gamma  # un-stretched current, for the physical diagnostics
    return SlabState(
        s_hat=T_hat - k * state.s_hat,
        displacement=s,
        wind_input=(tau.conjugate() * s).real,  # tau_x u + tau_y v
        damping_loss=rho * H * r * abs(s) ** 2,
        shear_exchange=rho * H * zeta * s.real * s.imag,
    )


def step_rk4(dt, s_n, tau_n, tau_np1, rho, H, f, r, zeta, forcing="linear"):
    """Advance the slab over one interval of length dt via 4th-order Runge-Kutta (integrators.rk4_step).

    Bundles the current with the parcel displacement, wind energy input, damping energy loss and
    front shear exchange into one state, and advances all five together through the same RK4 stages.

    Returns the same 5-tuple as step_exact.
    """
    gamma, omega = front_stretch(f, zeta)
    k = r + 1j * omega
    s_hat_n = s_n.real + 1j * gamma * s_n.imag

    rhs = partial(
        slab_rhs, tau_n=tau_n, tau_np1=tau_np1, dt=dt, rho=rho, H=H, r=r, zeta=zeta, gamma=gamma, k=k, forcing=forcing
    )
    state_0 = SlabState(s_hat=s_hat_n, displacement=0, wind_input=0, damping_loss=0, shear_exchange=0)
    result = rk4_step(rhs, 0.0, state_0, dt)

    s_next = result.s_hat.real + 1j * result.s_hat.imag / gamma
    return s_next, result.displacement, result.wind_input, result.damping_loss, result.shear_exchange



def integrate(t, tau, rho, H, f, r, zeta, s_0, step=step_exact, forcing="average"):
    """Run the slab through a wind stress time series, one interval at a time.

    Each step is handed the wind at both ends of its interval - what it does with that (hold it at
    the average, interpolate it, ...) is up to `step` - and advances the current, parcel position,
    wind energy input, damping energy loss and energy taken from the front's jet together, so all of
    them are integrated the same way.
    t: times [s], tau: wind stress tau_x + i tau_y at each time [N/m^2], zeta: background vorticity [1/s],
    s_0: velocity at t[0] [m/s]. forcing is passed straight through to `step` (both step_exact and
    step_rk4 accept "average" or "linear"; see either one's docstring).
    Returns s [m/s], position x + iy [m], and wind input, damping loss and shear exchange since t[0] [J/m^2].
    """
    s = np.zeros(len(t), dtype=complex)
    position = np.zeros(len(t), dtype=complex)
    wind_input = np.zeros(len(t))
    damping_loss = np.zeros(len(t))
    shear_exchange = np.zeros(len(t))
    s[0] = s_0
    for n, dt in enumerate(np.diff(t)):
        s[n + 1], d_position, d_wind_input, d_damping_loss, d_shear_exchange = step(
            dt, s[n], tau[n], tau[n + 1], rho, H, f, r, zeta, forcing=forcing
        )
        position[n + 1] = position[n] + d_position
        wind_input[n + 1] = wind_input[n] + d_wind_input
        damping_loss[n + 1] = damping_loss[n] + d_damping_loss
        shear_exchange[n + 1] = shear_exchange[n] + d_shear_exchange
    return s, position, wind_input, damping_loss, shear_exchange


STEPPERS = {"exact": step_exact, "rk4": step_rk4}  # config.step_type -> the function that implements it


def run(config):
    """Run the slab model for one configuration. Returns an xarray Dataset."""
    check_model_fields(config, "pollard_millard")  # uses no model-specific fields; warns about any given
    if config.step_type not in STEPPERS:
        raise ValueError(f"unknown step_type {config.step_type!r}: use {' or '.join(STEPPERS)}")
    t = np.arange(0, config.duration, config.dt)  # [s]
    tau = wind_from_config(t, config)  # wind stress [N/m^2]
    s_0 = config.u_initial + 1j * config.v_initial  # [m/s]
    s, position, wind_energy_input, damping_energy_loss, shear_energy_exchange = integrate(
        t, tau, config.rho, config.H, config.f, config.r, config.zeta, s_0,
        step=STEPPERS[config.step_type], forcing=config.forcing,
    )  # position: water parcel displacement [m]

    T = tau / (config.rho * config.H)  # forcing [m/s^2]
    # steady state for the wind at each time [m/s]: the Pollard-Millard one for the stretched current
    # s_hat = u + i gamma v (see step_exact), then unstretched
    gamma, omega = front_stretch(config.f, config.zeta)
    s_hat_steady = (T.real + 1j * gamma * T.imag) / (config.r + 1j * omega)
    s_steady = s_hat_steady.real + 1j * s_hat_steady.imag / gamma
    M = config.H * s  # depth-integrated transport [m^2/s]

    wind_work = (tau.conjugate() * s).real  # tau_x u + tau_y v, energy input from the wind [W/m^2]
    kinetic_energy = 0.5 * config.rho * config.H * np.abs(s) ** 2  # slab kinetic energy per unit area [J/m^2]
    damping_loss = config.rho * config.H * config.r * np.abs(s) ** 2  # energy removed by the damping [W/m^2]
    shear_production = config.rho * config.H * config.zeta * s.real * s.imag  # energy from the front's jet [W/m^2]
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
            "shear_production": (
                "time",
                shear_production,
                {"units": "W/m^2", "long_name": "energy taken from the front's jet, rho H zeta u v"},
            ),
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
            "shear_energy_exchange": (
                "time",
                shear_energy_exchange,
                {"units": "J/m^2", "long_name": "total energy taken from the front's jet since the start"},
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
