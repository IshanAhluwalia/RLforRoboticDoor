import os 
import torch as T
import torch.nn.functional as f
import numpy as np
from bufer import ReplayBuffer
from networks import *

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000,
                 layer1_size=256, layer2_size=128,
                 batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.noise = noise
        self.update_actor_iter = update_actor_interval

        # Create the networks
        self.actor = ActorNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='actor',
            learning_rate=actor_learning_rate
        )

        self.critic_1 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='critic_1',
            learning_rate=critic_learning_rate
        )

        self.critic_2 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='critic_2',
            learning_rate=critic_learning_rate
        )

        self.target_actor = ActorNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='target_actor',
            learning_rate=actor_learning_rate
        )

        self.target_critic_1 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='target_critic_1',
            learning_rate=critic_learning_rate
        )

        self.target_critic_2 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name='target_critic_2',
            learning_rate=critic_learning_rate
        )

    def choose_action(self, observation, validation: bool = False):
        """
        Returns a numpy action vector.
        - Warmup: random actions (uniform within bounds)
        - Otherwise: actor(state) + Gaussian noise (if not validation)
        """
        device = getattr(self.actor, "device", T.device("cpu"))

        # Prepare tensors
        state = T.as_tensor(observation, dtype=T.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # batch of 1

        min_a = T.as_tensor(self.min_action, dtype=T.float32, device=device)
        max_a = T.as_tensor(self.max_action, dtype=T.float32, device=device)

        with T.no_grad():
            # Warmup: random actions
            if (self.time_step < self.warmup) and (not validation):
                mu = T.as_tensor(
                    np.random.uniform(low=min_a.cpu().numpy(),
                                    high=max_a.cpu().numpy(),
                                    size=(self.n_actions,)),
                    dtype=T.float32, device=device
                )
            else:
                # Policy action
                mu = self.actor(state).squeeze(0)  # [n_actions]

            # Exploration noise (skip during validation)
            if not validation:
                noise = T.normal(mean=0.0, std=self.noise, size=mu.shape, device=device)
                mu = mu + noise

            # Clamp per-dimension to action bounds
            mu = T.max(T.min(mu, max_a), min_a)

        self.time_step += 1
        return mu.detach().cpu().numpy()


    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.store_transition(state, action, reward, next_state, done)


    def learn(self):
    # warmup: wait until we have enough samples
    if self.memory.mem_ctr < self.batch_size * 10:
        return

    # ----- sample a batch from replay buffer -----
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

    # ----- to tensors on the correct device -----
    reward     = T.as_tensor(reward, dtype=T.float32, device=device)
    done       = T.as_tensor(done,   dtype=T.float32, device=device)
    next_state = T.as_tensor(next_state, dtype=T.float32, device=device)
    state      = T.as_tensor(state,      dtype=T.float32, device=device)
    action     = T.as_tensor(action,     dtype=T.float32, device=device)

    # ----- target actions with clipped Gaussian noise (TD3 style) -----
    with T.no_grad():
        target_actions = self.target_actor(next_state)                      # π_target(s')
        noise = T.normal(mean=0.0, std=0.2, size=target_actions.shape, device=device)
        noise = T.clamp(noise, -0.5, 0.5)                                   # clip policy smoothing noise
        target_actions = target_actions + noise

        # clamp to action bounds (per-dimension)
        min_a = T.as_tensor(self.min_action, dtype=T.float32, device=device)
        max_a = T.as_tensor(self.max_action, dtype=T.float32, device=device)
        target_actions = T.max(T.min(target_actions, max_a), min_a)

        # next-state target Q-values
        next_q1 = self.target_critic_1(next_state, target_actions)
        next_q2 = self.target_critic_2(next_state, target_actions)

    # current Q estimates
    q1 = self.critic_1(state, action)
    q2 = self.critic_2(state, action)

        
    def update_network_parameters(self, tau):
        self.noise = noise 
        self.update_network_parameters(self, tau)
    
