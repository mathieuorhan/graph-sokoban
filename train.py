from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import gym_sokoban

from data.embedding import MinimalEmbedding, NoWallsEmbedding
from data.replay import ReplayMemory
from model.network import Net
from rl.explore import epsilon_greedy
from rl.optimize import optimize_model

# Parameters
TARGET_UPDATE = 5
BATCH_SIZE = 32
NUM_EPSIODES = 50
BUFFER_SIZE = 100
GAMMA = 0.999
RENDER_DISPLAY = "rgb_array"
ENV_LEVEL = "Sokoban-small-v1"
EPS = 0.1

# Create env
env = gym.make(ENV_LEVEL)

# Create buffer
memory = ReplayMemory(BUFFER_SIZE)

# GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create embedding
embedding = MinimalEmbedding()

# Create models
policy_net = Net(embedding.NUM_NODES_FEATURES)
target_net = Net(embedding.NUM_NODES_FEATURES)

# Create optimizer
optimizer = optim.RMSprop(policy_net.parameters())


for i_episode in range(NUM_EPSIODES):
    # Initialize the environment and state
    env.reset()

    state = embedding(env.render(embedding.RENDER_MODE))
    for t in count():
        # Select and perform an action
        action = epsilon_greedy(state, policy_net, EPS)
        _, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if done:
            next_state = None
        else:
            next_state = embedding(env.render(embedding.RENDER_MODE))

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if len(memory) >= BATCH_SIZE:
            optimize_model(
                policy_net=policy_net,
                optimizer=optimizer,
                memory=memory,
                gamma=GAMMA,
                batch_size=BATCH_SIZE,
            )

        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

