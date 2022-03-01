import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control  import rendering
from .Octocopter import Octocopter
from .Actuation import ControlAllocation
import pandas as pd
import numpy as np
import math

class OctorotorBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize the Octorotor
    # Simulation Parameters consist of the following and are passed via dictionary
    # g - gravity
    # mass - mass of octorotor
    # d - friction coefficient
    # b - rotor thrust constant
    # l - length of rotor arms (all equal)
    # omega_max - max angular velocity
    # Ixx - x moment of intertia
    # Iyy - y moment of inertia
    # Izz - z moment of inertia
    # dt  - time_Step of simulation
    # Motor - a motor object that inherits the Motor base class: Must have constructor and update functions implemented
    # motorController - A controller object that inherits the controller base class: Must have constructor and output methods implemented
    def __init__(self, OctorotorParams):
        super(OctorotorBaseEnv, self).__init__()
        # Octorotor Params
        self.octorotor = Octocopter(OctorotorParams)
        self.state = self.octorotor.get_state()
        self.xrefarr = pd.read_csv("../Paths/xrefZigtraj.csv", header=None).iloc[:, 1]
        self.yrefarr = pd.read_csv("../Paths/yrefZigtraj.csv", header=None).iloc[:, 1]
        self.allocation = ControlAllocation(OctorotorParams)
        self.faultVal = OctorotorParams["resistance"]
        self.dt = OctorotorParams["dt"]
        self.motor = OctorotorParams["motor"]
        self.motorController = OctorotorParams["motorController"]
        self.posc = OctorotorParams["positionController"]
        self.attc = OctorotorParams["attitudeController"]
        self.altc = OctorotorParams["altitudeController"]
        self.OctorotorParams = OctorotorParams
        self.omega = np.zeros(8)
        self.step_count = 0
        self.total_step_count = OctorotorParams["total_step_count"]
        self.zref = 0
        self.psiref = np.zeros(3)
        self.reward_discount = OctorotorParams["reward_discount"]
        # OpenAI Gym Params
        # State vector
        # state[0:2] pos
        # state[3:5] vel
        # state[6:8] angle
        # state[9:11] angle vel

        # poserrs+eulererrs
        # above + state
        # state 
        self.observation_space = spaces.Box(np.full(7, -1), np.full(7, 1))
        #U = [T, tau]
        #self.action_space = spaces.MultiDiscrete([3])
        self.action_space = spaces.Box(np.full(2, -1), np.full(2, 1))
        self.viewer = None
        self.init = False
        self.curTime = 0
        self.prevTime = 0
        self.traj = 0

    def dist(self, x, xref, y, yref):
        return math.sqrt((x-xref)**2 + (y-yref)**2)

    def stepSim(self):
        tau_des = self.attc.output(self.state, self.psiref)
        T_des = self.altc.output(self.state, self.zref)
        udes = np.array([T_des, tau_des[0], tau_des[1], tau_des[2]], dtype="float32")
        omega_ref = self.allocation.get_ref_velocity(udes)
        voltage = self.motorController.output(self.omega, omega_ref)
        self.omega = self.motor.update(voltage, self.dt)
        u = self.allocation.get_u(self.omega)
        self.octorotor.update_u(u)
        self.octorotor.update(self.dt)
        self.state = self.octorotor.get_state()

    def getWaypoint(self):
        self.curTime += 0.01
        if np.isclose(self.curTime-self.prevTime,0.5, 1e-9) and self.stepCount < len(self.xrefarr):
            self.prevTime = self.curTime
            self.xref = self.xrefarr[self.stepCount]
            self.yref = self.yrefarr[self.stepCount]
            self.stepCount += 1
            if self.stepCount == 5:
                self.motor.update_r(self.faultVal, self.brokenMotor)

    def step(self, action):
        errors = 0

        # USE THIS FOR BEFORE FAULT
        while(self.stepCount < 5):
            self.getWaypoint()
            targetValues = {"xref": self.xref, "yref": self.yref}
            self.psiref[1], self.psiref[0] = self.posc.output(self.state, targetValues)
            self.stepSim()
            self.errors = [self.xref-self.state[0], self.yref-self.state[1], self.zref-self.state[2]]
            errors += math.sqrt(self.errors[0]**2+self.errors[1]**2)

        # USE THIS LOOP FOR POST FAULT

        for i in range(10):
            self.getWaypoint()
            targetValues = {"xref": self.xref, "yref": self.yref}
            self.psiref[1], self.psiref[0] = self.posc.output(self.state, targetValues)
            
            # Controller impact
            if self.stepCount >= 5:
                self.psiref[1] += action[1]/5
                self.psiref[0] += action[0]/5
                min_ang = -12*math.pi/180
                max_ang = 12*math.pi/180
                self.psiref[1] = min(max(min_ang, self.psiref[1]), max_ang)
                self.psiref[0] = min(max(min_ang, self.psiref[0]), max_ang)

            self.stepSim()
            self.errors = [self.xref-self.state[0], self.yref-self.state[1], self.zref-self.state[2]]
            errors += math.sqrt(self.errors[0]**2+self.errors[1]**2)
        finish = self.terminate()
        self.errors = [self.xref-self.state[0], self.yref-self.state[1], self.zref-self.state[2]]
        self.eulererrors = [self.state[3] - self.psiref[0], self.state[4]-self.psiref[1], self.state[5]-self.psiref[2]]
        state = np.append(self.errors, self.eulererrors)
        state = np.append(state, self.loe[self.faultIndex])
        reward= -errors
        return state, reward, finish, {"xref": self.xref, "yref": self.yref, "x": self.state[0], "y": self.state[1], "motor": self.omega, "errors": errors, "traj":self.traj}

    def reset(self):
        # reset trajectories
        # ang velo is 700
        maxVal = 700
        self.loe = [0.8, 0.85, 0.9, 0.95, 1]
        self.brokenMotor = 0
        self.faultIndex = np.random.randint(0, 4)
        faultVal = maxVal * self.loe[self.faultIndex]

        self.traj = 0
        self.updateTraj()
        return self.resetNoTraj(faultVal)

    def resetNoTraj(self, faultVal):
        self.faultVal = faultVal
        self.init = False
        self.curTime = 0
        self.prevTime = 0
        self.stepCount = 1
        self.scaleFactor = 0
        OctorotorParams = self.OctorotorParams
        self.octorotor = Octocopter(OctorotorParams) 
        #self.octorotor.set_pos((b- a) * np.random.random_sample() + a, (b-a)*np.random.random_sample()+a)
        self.allocation = ControlAllocation(OctorotorParams)
        self.omega = np.zeros(8)
        self.dt = OctorotorParams["dt"]
        self.motor = OctorotorParams["motor"]
        self.motor.reset()
        self.motorController = OctorotorParams["motorController"]
        self.step_count = 0
        self.total_step_count = OctorotorParams["total_step_count"]
        self.viewer = None
        self.xref = self.xrefarr[0]
        self.yref = self.yrefarr[0]
        self.zref = 2
        self.index = 0
        self.psiref = np.zeros(3)
        self.state = self.octorotor.get_state()
        self.errors = [self.xref-self.state[0], self.yref-self.state[1], self.zref-self.state[2]]
        self.eulererrors = [self.state[3] - self.psiref[0], self.state[4]-self.psiref[1], self.state[5]-self.psiref[2]]
        state = np.append(self.errors, self.eulererrors)
        state = np.append(state, self.loe[self.faultIndex])
        error = math.sqrt((self.xref-self.state[0])*(self.xref-self.state[0]) + (self.yref-self.state[1]) * (self.yref-self.state[1]))
        return state

    def render(self,mode='human'):
        xref = self.xref
        yref = self.yref
        screen_width = 600
        screen_height = 600
        # Set width to 100x100
        world_width = 600
        scale = screen_width/world_width
        rotorradius = 4
        armwidth = 1
        armlength = self.OctorotorParams["l"]*scale + rotorradius
        if self.viewer is None:
            # build Octorotor
            self.viewer = rendering.Viewer(screen_width, screen_height)
            rotor = rendering.make_circle(radius=rotorradius)
            self.rotortrans = rendering.Transform()
            rotor.add_attr(self.rotortrans)
            rotor.set_color(1, 0, 0)
            self.viewer.add_geom(rotor)
            self.add_arm((0, 0), (armlength, 0))
            self.add_arm((0, 0), (-armlength, 0))
            self.add_arm((0, 0), (0, armlength))
            self.add_arm((0, 0), (0, -armlength))
            self.add_arm((0, 0), (armlength, armlength))
            self.add_arm((0, 0), (-armlength, armlength))
            self.add_arm((0, 0), (-armlength, -armlength))
            self.add_arm((0, 0), (armlength, -armlength))
            # Build ref Point
            refPoint = rendering.make_circle(radius = rotorradius)
            self.refPointTrans = rendering.Transform()
            refPoint.add_attr(self.refPointTrans)
            refPoint.set_color(0, 0, 1)
            self.refPointTrans.set_translation(xref*scale+screen_width/2, yref*scale+screen_width/2)
            self.viewer.add_geom(refPoint)
            
        if self.state is None:
            return None
        # Translate Rotor according to x, y
        x = self.state[0]
        y = self.state[1]
        rotorx = x*scale + screen_width/2.0
        rotory = y*scale + screen_width/2.0
        self.rotortrans.set_translation(rotorx, rotory)
        self.refPointTrans.set_translation(xref*scale+screen_width/2, yref*scale+screen_width/2)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def add_arm(self, start, end):
        arm = rendering.Line(start=start, end=end)
        arm.add_attr(self.rotortrans)
        self.viewer.add_geom(arm)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state(self):
        return self.state

    def get_xerror(self):
        return self.state[0]

    def get_yerror(self):
        return self.state[1]

    def updateTraj(self):
        if self.traj == 0:
            self.xrefarr = pd.read_csv("../Paths/xrefZigtraj.csv", header=None).iloc[:, 1]
            self.yrefarr = pd.read_csv("../Paths/yrefZigtraj.csv", header=None).iloc[:, 1]
        elif self.traj == 1:
            self.xrefarr = pd.read_csv("../Paths/xrefCirctraj.csv", header=None).iloc[:, 1]
            self.yrefarr = pd.read_csv("../Paths/yrefCirctraj.csv", header=None).iloc[:, 1]
        elif self.traj == 2:
            self.xrefarr = pd.read_csv("../Paths/xrefEtraj.csv", header=None).iloc[:, 1]
            self.yrefarr = pd.read_csv("../Paths/yrefEtraj.csv", header=None).iloc[:, 1]

    def terminate(self):
        #dist = self.dist(self.state[0], self.finalx, self.state[1], self.finaly)
        #if dist < 1:
            #return True
        if self.traj == 0 and self.curTime > 43:
            self.traj = 1
            self.resetNoTraj(self.faultVal)
            self.updateTraj()
            return False
        elif self.traj == 1 and self.curTime > 50:
            self.traj = 2
            self.resetNoTraj(self.faultVal)
            self.updateTraj()
            return False
        elif self.traj == 2 and self.curTime > 45:
            return True
        return False

