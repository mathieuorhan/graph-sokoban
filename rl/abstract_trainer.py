import random

from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from data.replay import ReplayMemory


class AbstractTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.seed_experiment()
        self.device = self.get_device()
        self.embedding = opt.embedding
        self.build_env()
        self.build_networks()
        self.build_optimizer()
        self.build_datasets()
        self.build_memory()
        self.build_scheduler()
        self.epoch = 0
        self.episodes_seen = 0
        self.update_count = 0
        self.display_parameters()

    def display_parameters(self):
        for key, value in self.opt.items():
            print(f"{colored(key, 'blue')}: {value}")

    def build_memory(self):
        self.memory = ReplayMemory(self.opt.buffer_size)

    def build_env(self):
        raise NotImplementedError

    def build_networks(self):
        raise NotImplementedError

    def build_datasets(self):
        raise NotImplementedError

    def build_scheduler(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.opt.lr,
            alpha=self.opt.rms_alpha,
            eps=self.opt.rms_eps,
        )

    def seed_experiment(self):
        np.random.seed(self.opt.seed)
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

