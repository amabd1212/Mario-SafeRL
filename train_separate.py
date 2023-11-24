import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from SeparateNN import Actor,Critic  # Adjusted import for the actor model

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

from pathlib import Path
from metric_logger import MetricLogger

import math,random,datetime


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


    # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v2", render_mode='human', apply_api_compatibility=True)


env = JoypadSpace(env,  SIMPLE_MOVEMENT)

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

# Initialize MetricLogger
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)# Replace with your actual directory
logger = MetricLogger(save_dir)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99        # Discount factor
num_episodes = 40000 # Total number of episodes

epsilon_start = 0.6
epsilon_final = 0.01
epsilon_decay = 50000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# Environment setup
num_actions = len(SIMPLE_MOVEMENT)  # Number of possible actions
num_inputs = 4  # Number of channels (stacked frames)

# Initialize actor and critic models
actor_model = Actor(num_inputs, num_actions)
critic_model = Critic(num_inputs)

# Separate optimizers for actor and critic
actor_optimizer = optim.Adam(actor_model.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=learning_rate)

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


for episode in range(num_episodes):
    state = env.reset()
    state = first_if_tuple(state)
    state = np.array(state)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logger.init_episode()

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    for steps in range(10000): 
        epsilon = epsilon_by_frame(episode * 100 + steps)
        policy = actor_model(state)
        value = critic_model(state)
        m = Categorical(policy)
        if random.random() > epsilon:
            # Follow the policy
            action = policy.argmax()
        else:
        #     # Take a random action
            action = torch.tensor([env.action_space.sample()], dtype=torch.int)
        next_state, reward, done, trun, info = env.step(action.item())
        # Process next state and compute losses
        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        log_prob = m.log_prob(action)
        entropy += m.entropy().mean()
        # print(log_prob,reward,entropy)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1-done], dtype=torch.float))
        print(action, log_prob)
        state = next_state
        logger.log_step(reward)
        if info['flag_get'] == True:
            print("YOOOOOOO")
        if done:
            break
    next_state = torch.FloatTensor(next_state)
    next_value = critic_model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    # Update actor
    actor_loss = -(log_probs * advantage.detach()).mean()  -  entropy
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Update critic
    critic_loss = advantage.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    logger.log_episode(critic_loss.item())
    if episode % 1000 == 0:
        print('Episode {}: Last reward: {}'.format(episode, reward))
        logger.record(episode, steps)

env.close()
