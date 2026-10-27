"""Physical relationships"""

import numpy as np

OMEGA=7.2921e-5 # Rotation rate of earth [rad/s]

def coriolis_parameter(lat_degree):
    return 2 * OMEGA * np.sin(lat_degree*np.pi/180)