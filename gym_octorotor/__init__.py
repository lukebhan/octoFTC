from gym.envs.registration import register

register(
        id='octorotor-v0',
        entry_point='gym_octorotor.envs:OctorotorBaseEnv'
)
