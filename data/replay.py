from .constants import Transition
from .graph_env import GraphEnv
import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a Transition with batched values."""
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class RewardReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.total_reward = 0.0
        # Constant to add to each reward to make it positive
        self.add_reward = 0.1 - min(
            [
                GraphEnv.STEP_REWARD,
                GraphEnv.ON_BOX_REWARD,
                GraphEnv.OFF_BOX_REWARD,
                GraphEnv.FINISH_REWARD,
            ]
        )
        self.p = []

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        norm_reward = self.add_reward + reward

        if len(self.memory) < self.capacity:
            self.total_reward += norm_reward
            self.memory.append(Transition(state, action, next_state, reward))
            self.p.append(norm_reward)

        else:
            self.total_reward += norm_reward - self.p[self.position]
            self.memory[self.position] = Transition(state, action, next_state, reward)
            self.p[self.position] = norm_reward

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a Transition with batched values."""
        transitions_idx = np.random.choice(
            len(self), size=batch_size, p=[p / self.total_reward for p in self.p]
        )
        transitions = [self.memory[i] for i in transitions_idx]
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)
