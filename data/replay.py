from .constants import Transition
from .graph_env import GraphEnv
import random
import heapq
import numpy as np
from data.data_structure import MinSegmentTree, SegmentTree, SumSegmentTree
import torch


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


class GreedyPrioritizedExperienceReplay(ReplayMemory):
    """Greedy Prioritized replay memory using binary heap."""

    def __init__(self, capacity):
        super().__init__(capacity)

    def push(self, transition, TDerror):
        heapq.heappush(self.memory, (-TDerror, transition))
        if len(self.memory) < self.capacity:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def sample(self, batch_size):
        transitions = heapq.nsmallest(batch_size, self.memory)
        transitions = [t for (_, t) in transitions]
        self.memory = self.memory[batch_size:]
        return Transition(*zip(*transitions))


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


class ThresholdReplayMemory:
    """Replay memory where transitions whose reward is higher than a threshold are sampled a factor times more often."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.threshold = 0.0
        self.factor = 10.0
        self.n = 0
        self.p = []

    def push(self, state, action, next_state, reward):
        """Saves a transition."""

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward))
            if reward > self.threshold:
                self.n += 1
                self.p.append(self.factor)
            else:
                self.p.append(1.0)

        else:
            self.memory[self.position] = Transition(state, action, next_state, reward)
            if self.p[self.position] == self.factor:
                self.n -= 1
            if reward > self.threshold:
                self.n += 1
                self.p[self.position] = self.factor
            else:
                self.p[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a Transition with batched values."""
        transitions_idx = np.random.choice(
            len(self),
            size=batch_size,
            p=[p / (self.n * (self.factor - 1) + len(self)) for p in self.p],
        )
        transitions = [self.memory[i] for i in transitions_idx]
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)
