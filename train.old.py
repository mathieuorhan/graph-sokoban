from itertools import count
import warnings
from tqdm import tqdm

# Ignoring warnings for Sokoban level generation
# hardcoded print
# See https://github.com/mpSchrader/gym-sokoban/blob/bb764792bb0f43c647d636cb395286276c82fe70/gym_sokoban/envs/sokoban_env.py
# warnings.filterwarnings("ignore", message="Not enough free spots")
# warnings.filterwarnings("ignore", message="Generated Model with score == 0")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import gym_sokoban

import numpy as np
import random

from data.embedding import MinimalEmbedding, NoWallsEmbedding
from data.replay import ReplayMemory
from model.network import Net
from rl.explore import epsilon_greedy
from rl.optimize import optimize_model

# Parameters
TARGET_UPDATE = 5
GAMMA = 1.0
RENDER_DISPLAY = "rgb_array"
ENV_LEVEL = "Sokoban-small-v0"
EPS_0 = 0.8
EPS_DECAY = 0.9
BATCH_SIZE = 32
BUFFER_SIZE = 128
NUM_EPISODES = 200
NUM_EPISODES_EVAL = 20
MAX_STEPS = 25
MAX_STEPS_EVAL = 25
NUM_EPOCHS = 50
STATIC_LEVEL = False
SEED = 123

# Seed
np.random.seed(SEED)
random.seed(SEED)

# Create env
env = gym.make(ENV_LEVEL)

# Create buffer
memory = ReplayMemory(BUFFER_SIZE)

# GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create embedding
embedding = NoWallsEmbedding()

# Create models
policy_net = Net(embedding.NUM_NODES_FEATURES).to(device)
target_net = Net(embedding.NUM_NODES_FEATURES).to(device)

# Create optimizer
optimizer = optim.RMSprop(policy_net.parameters())

# Epislon Greedy random number generator
random_generator = random.Random()
eps = EPS_0

for epoch in range(NUM_EPOCHS):
    policy_net.train()
    mean_total_reward = 0
    for episode in range(NUM_EPISODES):
        # Initialize the environment and state
        if STATIC_LEVEL:
            # Issue : neutrize epislon greedy !
            np.random.seed(SEED)
            random.seed(SEED)
        env.reset()
        total_reward = 0.0
        state = embedding(env.render(embedding.RENDER_MODE))
        for t in range(MAX_STEPS):
            # Select and perform an action
            action_node, action = epsilon_greedy(
                state, policy_net, eps, random_generator, 0.05, 0.025
            )
            _, reward, done, _ = env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None
            else:
                next_state = embedding(env.render(embedding.RENDER_MODE))

            # Store the transition in memory
            memory.push(state, action_node, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(
                target_net=target_net,
                policy_net=policy_net,
                optimizer=optimizer,
                memory=memory,
                gamma=GAMMA,
                batch_size=BATCH_SIZE,
            )

            if done:
                break

        # Update the target network, copying all weights and biases in DQN
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        mean_total_reward += total_reward / NUM_EPISODES
    eps *= EPS_DECAY
    print(
        "[Training]   epoch %2d Mean reward per episode: %.2f. (eps=%.2f)"
        % (epoch, mean_total_reward, eps)
    )

    # Evaluation
    policy_net.eval()
    mean_total_reward = 0
    for episode in range(NUM_EPISODES_EVAL):
        # Initialize the environment and state
        env.reset()
        total_reward = 0.0

        for t in range(MAX_STEPS_EVAL):
            state = embedding(env.render(embedding.RENDER_MODE))
            # Select and perform an action
            _, action = epsilon_greedy(state, policy_net, 0, None)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            # Observe new state
            if done:
                print("BOOOOOM")
                break

        mean_total_reward += total_reward / NUM_EPISODES_EVAL
    print(
        "[Evaluation] epoch %2d Mean reward per episode: %.2f"
        % (epoch, mean_total_reward)
    )

    # Save model
    torch.save(policy_net.state_dict(), "test.pth")

