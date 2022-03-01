# Brushless DC motor
# Inherits the Motor based class
# Defines the dynamics of a standard dc motor

import numpy as np
import scipy.integrate 
from .Motor import Motor
from numba import jit

@jit
def f(km, R, v, d, omega, ke, Izzm):
    res = []
    for i in range(8):
        res.append((km/R[i]*v[i]-d*omega[i]*omega[i] -km*ke /R[i] * omega[i])/Izzm)
    return res


class BLDCM(Motor):
    # Takes in as parameters:
    # km - mehcanical motor constant
    # ke - electrical motor constant
    # R - resistance
    # d - the modelled drage momment
    # Izzm - moment of inertia about z for a motor
    def __init__(self, motorArgs):
        self.motorArgs = motorArgs
        self.km = np.float32(motorArgs["km"])
        self.ke = np.float32(motorArgs["ke"])
        self.R = np.full(8, np.float32(motorArgs["R"]))
        self.d = np.float32(motorArgs["d"])
        self.Izzm = np.float32(motorArgs["Izzm"])
        self.stepNum = 0
        self.omega = np.zeros(8, dtype="float32")
        self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode', method='bdf')
        self.cap = np.full(8, np.inf)

    def reset(self):
        motorArgs = self.motorArgs
        self.km = np.float32(motorArgs["km"])
        self.ke = np.float32(motorArgs["ke"])
        self.R = np.full(8, np.float32(motorArgs["R"]))
        self.d = np.float32(motorArgs["d"])
        self.Izzm = np.float32(motorArgs["Izzm"])
        self.stepNum = 0
        self.omega = np.zeros(8, dtype="float32")
        self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode', method='bdf')
        self.cap = np.full(8, np.inf)

    # Return a motors angular velocity moving one step in time with a given voltage
    def update(self, voltage, dt):
        self.stepNum += 1
        self.v = voltage
        self.ode.set_initial_value(self.omega, 0)
        self.omega = self.ode.integrate(self.ode.t + dt)
        for i in range(len(self.omega)):
            self.omega[i] = min(self.cap[i], self.omega[i])
        return self.omega

    # Helper Method to calculate omega_dot for our ode integrator. Can be written as a lambda function inside update for other shorter motors.
    def omega_dot_i(self, time, state):
        res = f(self.km, self.R, self.v, self.d, self.omega, self.ke, self.Izzm)
        return res

    def update_r(self, r, idx):
        self.cap[idx] = r
