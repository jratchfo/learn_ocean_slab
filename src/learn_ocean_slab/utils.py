import math
import numpy as np

def vec_velocity(u,v):
    #returns complex number
    return u+v*(0+1j)

def sca_velocity(vec):
    #returns two scalars
    return np.real(vec), np.imag(vec)