import time 
import os
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite 
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from networks import *
from buffer import ReplayBuffer


if __name__ == '__main__':
    if not os.path.exists("tmp/robo"):
        os.makedirs("tmp/robo")

    env_name = "Door"

    config = load_composite_controller_config(controller="BASIC", robot="PANDA")
    config["body_parts"]["right"]["type"] = "JOINT_VELOCITY"
    
    env = suite.make(
        env_name,
        robots = ["Panda"],
        controller_configs = config,
        has_renderer = False,
        use_camera_obs = False, 
        horizon = 300,
        reward_shaping = True,
        control_freq = 20,
    )

    env = GymWrapper(env)

'''
    critic_network = CriticNetwork([8], 8)
    actor_network = ActorNetwork([8], 8)

    replay_buffer = ReplayBuffer(8, [8], 8)
'''

