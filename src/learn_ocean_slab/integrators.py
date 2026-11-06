"""Generic ODE steppers, model-agnostic: given a right-hand-side function, advance one time step.
"""


def rk4_step(rhs, t, s, dt):
    """4th-order Runge-Kutta: advance s from t to t + dt under ds/dt = rhs(t, s).
    """
    k1 = rhs(t, s)
    k2 = rhs(t + dt / 2, s + dt / 2 * k1)
    k3 = rhs(t + dt / 2, s + dt / 2 * k2)
    k4 = rhs(t + dt, s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
