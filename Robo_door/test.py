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
from robo_torch import Agent

if __name__ == '__main__':

    if not os.path.exists("tmp/robo"):
        os.makedirs("tmp/robo")

    env_name = "Door"

    config = load_composite_controller_config(controller="BASIC", robot="PANDA")
    
    # Fix the gripper action dimensions for JOINT_VELOCITY to match your training
    if "body_parts" in config and "right" in config["body_parts"]:
        config["body_parts"]["right"]["type"] = "JOINT_VELOCITY"
        config["body_parts"]["right"]["input_max"] = 1.0
        config["body_parts"]["right"]["input_min"] = -1.0
        # Fix output dimensions for 7 joint velocities
        config["body_parts"]["right"]["output_max"] = [1.0] * 7
        config["body_parts"]["right"]["output_min"] = [-1.0] * 7
    
    env = suite.make(
        env_name,
        robots = ["Panda"],
        controller_configs = config,
        has_renderer = True,
        has_offscreen_renderer = False,
        use_camera_obs = False, 
        horizon = 300,
        reward_shaping = True,
        control_freq = 20,
        render_camera = None,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 100
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate = actor_learning_rate, 
                critic_learning_rate = critic_learning_rate, 
                tau = 0.005, 
                input_dims=env.observation_space.shape, 
                env=env,
                n_actions=env.action_space.shape[0],
                layer_1_size=layer_1_size,
                layer_2_size=layer_2_size,
                batch_size=batch_size,
                noise=0.1)

    n_games = 3
    agent.load_models()
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate = {critic_learning_rate}"

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation, validation = True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            env.render()  # Disabled due to macOS mjpython requirement
            time.sleep(0.05)
            done = terminated or truncated
            score += reward
            observation = next_observation


        print(f"episode: {i} score: {score}")


