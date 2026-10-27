''' 
Pollard-Millard Slab model (based on their 1970 paper)
'''
import numpy as np
import xarray as xr
from .utils import *
from .forcing import wind_from_config

def exp_ix(x):
    return np.exp(1j*x)
    #return np.cos(x)+ np.sin(x)*(0+1j) # the same thing

def pm_model_current_response_constant_wind(t, tau_over_rho_h, f, r,  s_0):
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
    return s_0*(exp_ix(-f*t))*np.exp(-r*t) + tau_over_rho_h/(f*(1j)+r) * (1- exp_ix(-f*t)*np.exp(-r*t))

def pm_model_instant_transport_constant_wind(s, H):
    """ depth integrated transport [m^2/s] - since nothing depends on Z,  int( s(t) dz) = H s(t)"""
    return H *s 


def pm_model_integrated_transport_constant_wind(s, H):
    M = H *s 




def pm_model_integrated_transport_constant_wind(t, tau_over_rho_h, f, r, s_0, H):
    """Transport accumulated since t=0, int_0^t M dt' [m^2], i.e. m^3 of water per m of width.

    - From the current solution, with s_p = T/(if+r):
    s(t') = s_p + (s_0 - s_p) exp(-(if+r)*t')

    - Integrate each term, using int_0^t exp(-(if+r)*t') dt' = (1 - exp(-(if+r)*t))/(if+r):
    int_0^t s dt' = s_p*t + (s_0 - s_p) * (1 - exp(-(if+r)*t))/(if+r)

    - Multiply by H to go from velocity to transport.
    Divided by H, this is the displacement of a water parcel in the slab [m].
    """
    s_p = tau_over_rho_h / (1j*f + r)
    return H * (s_p*t + (s_0 - s_p) * (1 - np.exp(-(1j*f + r)*t)) / (1j*f + r))


def pm_model_current_response(t, tau, rho, H, f, r, s_0):
    """Pollard-Millard slab current driven by a wind stress time series.

    t: times [s] (array), tau: wind stress tau_x + i tau_y at each time [N/m^2] (array),
    rho: density [kg/m^3], H: slab depth [m], f: Coriolis [1/s], r: damping rate [1/s],
    s_0: velocity at t[0] [m/s].
    Returns complex velocity s = u + iv at each time [m/s].
    """
    s = np.zeros(len(t), dtype=complex)
    s[0] = s_0
    d_tau = np.diff(tau)
    for n, dt in enumerate(np.diff(t),0):
        T = (tau[n+1]-d_tau[n]/2 )/(rho*H)       
        s[n+1] = pm_model_current_response_constant_wind(dt,T,f,r,s[n] )
    
    return s


def pm_model_position(t, tau, rho, H, f, r, s):
    """Water parcel displacement from its starting point, driven by a wind stress time series.

    Uses the same interval-averaged forcing as pm_model_current_response, and the exact
    displacement over each interval from pm_model_integrated_transport_constant_wind.
    t: times [s], tau: wind stress [N/m^2], s: velocity from pm_model_current_response [m/s].
    Returns complex displacement x + iy at each time [m].
    """
    position = np.zeros(len(t), dtype=complex)
    d_tau = np.diff(tau)
    for n, dt in enumerate(np.diff(t),0):
        T = (tau[n+1]-d_tau[n]/2 )/(rho*H)   
        position[n+1] = position[n] + pm_model_integrated_transport_constant_wind(dt, T, f, r, s[n], H) / H
    return position

def run(config):
    """Run the slab model for one configuration. Returns an xarray Dataset."""
    t = np.arange(0, config.duration, config.dt)  # [s]
    tau = wind_from_config(t, config)  # wind stress [N/m^2]
    s_0 = config.u_initial + 1j * config.v_initial  # [m/s]
    s = pm_model_current_response(t, tau, config.rho, config.H, config.f, config.r, s_0)

    T = tau / (config.rho * config.H)  # forcing [m/s^2]
    s_steady = T / (1j * config.f + config.r)  # steady state for the wind at each time [m/s]
    M = config.H * s  # depth-integrated transport [m^2/s]

    wind_work = (tau.conjugate() * s).real  # tau_x u + tau_y v, energy input from the wind [W/m^2]
    angle_to_wind = np.where(tau != 0, np.degrees(np.angle(s * tau.conjugate())), np.nan)  # current direction minus wind direction [degrees]; NaN with no wind
    position = pm_model_position(t, tau, config.rho, config.H, config.f, config.r, s) #water parcel position

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
        "angle_to_wind": ("time", angle_to_wind, {"units": "degrees", "long_name": "current direction relative to the wind, positive = left of the wind"}),
        "x": ("time", position.real, {"units": "m", "long_name": "eastward parcel displacement"}),
        "y": ("time", position.imag, {"units": "m", "long_name": "northward parcel displacement"})
        },
        coords={"time": ("time", t, {"units": "s"})},
        attrs=config.to_dict(),
    )