from .constants import Transition
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

