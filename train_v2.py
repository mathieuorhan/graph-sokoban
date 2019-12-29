from tqdm import tqdm

from torch_geometric.nn import GraphUNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import random

from data.dataset import InMemorySokobanDataset
from data.embedding import MinimalEmbedding, NoWallsEmbedding, NoWallsV2Embedding
from data.replay import ReplayMemory
from data.graph_env import GraphEnv
from model.network import Net
from rl.explore import epsilon_greedy_only_graph, best_from_nodes
from rl.optimize import optimize_model
from rl.schedulers import AnnealingScheduler

# Data parameters
TRAIN_PATH = "levels/dummy_small_100"
TEST_PATH = "levels/dummy_small_100"

# RL parameters
TARGET_UPDATE = 20
GAMMA = 1.0
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_STOP_STEP = 3000
BATCH_SIZE = 32
BUFFER_SIZE = 5000
MAX_STEPS = 10
MAX_STEPS_EVAL = 10
NUM_EPOCHS = 200
SEED = 123
WALLS_PROBS = 0
STATIC_PROBS = 0

# Deadlocks parameters
EARLY_STOP_DEADLOCKS = True
PENALIZE_DEADLOCKS = True
REWARD_DEADLOCKS = -1
GO_BACK_AFTER_DEADLOCKS = False

# Opt parameters
LEARNING_RATE = 0.00025
RMS_ALPHA = 0.95
RMS_EPS = 0.01

# Model parameters
HIDDEN_CHANNELS = 64
DEPTH = 4
ACT = F.relu
POOL_RATIOS = 0.5
SUM_RES = False

# "good results": hidden = 64, depth = 3, pool = 0.5, sum_res = 1
# sum_res = False seems better


# Seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Create buffer
memory = ReplayMemory(BUFFER_SIZE)

# GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create embedding
embedding = NoWallsV2Embedding()

# Create models
# policy_net = Net(embedding.NUM_NODES_FEATURES).to(device)
# target_net = Net(embedding.NUM_NODES_FEATURES).to(device)
policy_net = GraphUNet(
    in_channels=embedding.NUM_NODES_FEATURES,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=1,
    depth=DEPTH,
    pool_ratios=POOL_RATIOS,
    sum_res=SUM_RES,
    act=ACT,
).to(device)

target_net = GraphUNet(
    in_channels=embedding.NUM_NODES_FEATURES,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=1,
    depth=DEPTH,
    pool_ratios=POOL_RATIOS,
    sum_res=SUM_RES,
    act=ACT,
).to(device)


# Create optimizer
optimizer = optim.RMSprop(
    policy_net.parameters(), lr=LEARNING_RATE, alpha=RMS_ALPHA, eps=RMS_EPS
)

# Create datasets
dataset_train = InMemorySokobanDataset(TRAIN_PATH, embedding, device=device)
dataset_test = InMemorySokobanDataset(TEST_PATH, embedding, device=device)

# GraphEnv
env = GraphEnv()

# Epislon Greedy andrandom number generator
random_generator = random.Random()
scheduler = AnnealingScheduler(EPS_MAX, EPS_MIN, EPS_STOP_STEP)

episodes_seen = 0

for epoch in range(NUM_EPOCHS):
    print(f"=== EPOCH {epoch} [eps={scheduler.epsilon:.2f}] ===")
    policy_net.train()
    mean_total_reward = 0
    total_solved = 0
    total_deadlocks = 0
    # Sample the episodes
    episode_indexes = list(range(len(dataset_train)))
    random.shuffle(episode_indexes)
    for episode_idx in episode_indexes:
        # Initialize the environment and state
        env.reset(dataset_train[episode_idx])
        total_reward = 0.0

        for t in range(MAX_STEPS):
            state = env.render()
            # Select and perform an action
            with torch.no_grad():
                action_node = epsilon_greedy_only_graph(
                    state,
                    policy_net,
                    scheduler.epsilon,
                    random_generator,
                    WALLS_PROBS,
                    STATIC_PROBS,
                )
                next_state, reward, done, info = env.step(action_node)
            if info["deadlock"] and PENALIZE_DEADLOCKS:
                reward += REWARD_DEADLOCKS
            total_reward += reward
            reward = torch.tensor(reward, device=device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action_node, next_state, reward)

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
                total_solved += 1
                break

            if info["deadlock"]:
                total_deadlocks += 1
                if EARLY_STOP_DEADLOCKS:
                    break
                if GO_BACK_AFTER_DEADLOCKS:
                    # Go back directly to previous state
                    env.state = state

        # Update the target network, copying all weights and biases in DQN
        episodes_seen += 1
        if episodes_seen % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        mean_total_reward += total_reward / len(dataset_train)
        scheduler.step()
    mean_solved = total_solved / len(dataset_train)
    print(
        f"[Train] Mean reward: {mean_total_reward:.2f}. Solved: {total_solved}/{len(dataset_train)} ({100*mean_solved:.2f}%). Deadlocks: {total_deadlocks}/{len(dataset_train)}"
    )

    # Evaluation
    policy_net.eval()
    total_solved = 0
    mean_total_reward = 0
    for init_state in dataset_test:
        # Initialize the environment and state
        env.reset(init_state)
        total_reward = 0.0
        for t in range(MAX_STEPS_EVAL):
            state = env.render()
            # Select and perform an action
            with torch.no_grad():
                scores = policy_net(state.x, state.edge_index)
                action_node = best_from_nodes(
                    scores, state, WALLS_PROBS == 0, STATIC_PROBS == 0
                )
                next_state, reward, done, _ = env.step(action_node)
            # Observe new state
            if done:
                total_solved += 1
                break
            else:
                total_reward += reward

        mean_total_reward += total_reward / len(dataset_test)
    mean_solved = total_solved / len(dataset_test)
    print(
        f"[Eval.] Mean reward: {mean_total_reward:.2f}. Solved: {total_solved}/{len(dataset_test)} ({100*mean_solved:.2f}%)"
    )

# Save model
torch.save(policy_net.state_dict(), "test.pth")

