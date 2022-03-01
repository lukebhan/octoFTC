import gym
import time
from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gym_octorotor
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math


# Simulation Parameters
g = 9.81
m0 = 2
d = 1.36787E-7
Ixx = 0.0429
Iyy = 0.0429
Izz = 0.0748

OctorotorParams = {
        "g": g,
        "m0": m0,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "b": 8.54858E-6,
        "d": d,
        "l": 1,
        "omegamax": 600000,
        "dt": 0.01
}

MotorParams = {
        "Izzm": 2E-5,
        "km": 0.0107,
        "ke": 0.0107,
        "R": 0.2371,
        "d": d,
        "komega": 2,
        "maxv": 11.1
}

PositionParams = {
        "kpx": 0.5,
        "kdx":  .1,
        "kpy": 0.5,
        "kdy": .1,
        "min_angle": -12*math.pi/180,
        "max_angle": 12*math.pi/180
}
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

AttitudeParams = {
        "kd": 10,
        "kp": 50,
        "j" : J
}

AltitudeParams = {
        "g": g,
        "m0": m0,
        "kdz": 24,
        "kpz": 144
}

# Setup PID Controllers
altc = AltitudeController(AltitudeParams)
attc = AttitudeController(AttitudeParams)
posc = PositionController(PositionParams)

# Create Motor and motor controllers
motor = BLDCM(MotorParams)
motorc = MotorController(MotorParams)
OctorotorParams["motor"] = motor
OctorotorParams["motorController"] = motorc
OctorotorParams["positionController"] = posc
OctorotorParams["attitudeController"] = attc
OctorotorParams["altitudeController"] = altc
OctorotorParams["total_step_count"] = 5000
OctorotorParams["reward_discount"] = 1
resistance = np.full(8, 0.2371)
OctorotorParams["resistance"] = resistance
env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
# inner function

def err(state):
    return math.sqrt(state[0]**2 + state[1]**2)

def run(faultVal):
    t1xarr = []
    t1yarr = []
    t1running_err = 0

    t2xarr = []
    t2yarr = []
    t2running_err = 0

    t3xarr = []
    t3yarr = []
    t3running_err = 0

    model = PPO.load('model_no_fault', env=env)

    end =False 
    obs  = env.reset(faultVal)
    #obs  = env.reset()
    action = [0, 0]
    count = 0
    reward = 0
    traj = 0
    while not end:
        action = model.predict(obs, deterministic=True)[0]
        state, rew, end, prints = env.step(action)
        traj = prints["traj"]
        if traj == 0:
            t1running_err += prints["errors"]
            t1xarr.append(prints["x"])
            t1yarr.append(prints["y"])
        elif traj == 1:
            t2running_err += prints["errors"]
            t2xarr.append(prints["x"])
            t2yarr.append(prints["y"])
        elif traj == 2:
            t3running_err += prints["errors"]
            t3xarr.append(prints["x"])
            t3yarr.append(prints["y"])
        count += 1
        reward += rew
    print(count)
    print(reward)
    return t1running_err, t1xarr, t1yarr, t2running_err, t2xarr, t2yarr, t3running_err, t3xarr, t3yarr

def save2darr(arr, filename):
    f = open(filename, 'w')
    for i in range(len(arr[0])):
        for j in range(len(arr)):
            f.write(str(arr[j][i]) + " " )
        f.write("\n")
    f.close()

fault_val = []
t1x = []
t1y = []
t1e = []
t2x = []
t2y = []
t2e = []
t3x = []
t3y = []
t3e = []
rpmarr = []
if __name__ == "__main__":
    val = 0
    for i in range(2):
        print(i)
        t1running, t1xval, t1yval, t2running, t2xval, t2yval, t3running, t3xval, t3yval = run(val)
        fault_val.append(val)
        t1x.append(t1xval)
        t1y.append(t1yval)
        t2x.append(t2xval)
        t2y.append(t2yval)
        t3x.append(t3xval)
        t3y.append(t3yval)
        t1e.append(t1running)
        t2e.append(t2running)
        t3e.append(t3running)
    save2darr(t1x, "t1x")
    save2darr(t1y, "t1y")
    np.savetxt("t1error", t1e)

    save2darr(t2x, "t2x")
    save2darr(t2y, "t2y")
    np.savetxt("t2error", t2e)

    save2darr(t3x, "t3x")
    save2darr(t3y, "t3y")
    np.savetxt("t3error", t3e)
