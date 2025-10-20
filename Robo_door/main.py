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
    
    writer = SummaryWriter(f'logs/PPO')
    n_games = 10000
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate = {critic_learning_rate}"

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
        
        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if i % 10 == 0:
            agent.save_models()
        
        print(f"episode: {i} score: {score}")

