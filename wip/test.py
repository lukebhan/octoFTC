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

def run():
    xarr = []
    yarr = []
    xrefarr = []
    yrefarr = []
    rpm = []
    tau = []
    end =False 
    model = PPO.load("tmp/model_40000_steps", env=env)
    obs  = env.reset()
    #obs  = env.reset()
    action = [0]
    running_err = 0
    count = 0
    reward = 0
    while not end:
        action = model.predict(obs)[0]
        state, rew, end, prints = env.step(action)
        running_err += prints["errors"]
        xarr.append(prints["x"])
        yarr.append(prints["y"])
        rpm.append(prints["motor"][2])
        count += 1
        reward += rew
    print(count)
    print(reward)
    return xarr, yarr, running_err, rpm

def save2darr(arr, filename):
    f = open(filename, 'w')
    for i in range(len(arr[0])):
        for j in range(len(arr)):
            f.write(str(arr[j][i]) + " " )
        f.write("\n")
    f.close()

x_runs = []
y_runs = []
fault_val = []
errors = []
rpmarr = []
if __name__ == "__main__":
    for i in range(2):
        print(i)
        xtmp, ytmp, errtmp, rpm= run()
        x_runs.append(xtmp)
        y_runs.append(ytmp)
        errors.append(errtmp)

    save2darr(x_runs, "x_runs")
    save2darr(y_runs, "y_runs")
    np.savetxt("errors", errors)
