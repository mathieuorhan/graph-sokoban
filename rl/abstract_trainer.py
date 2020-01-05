import random
import logging

from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from data.replay import ReplayMemory, RewardReplayMemory, ThresholdReplayMemory


class AbstractTrainer:
    def __init__(self, opt):
        self.logger = logging.getLogger()
        self.info = {}
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
        self.display_info()

    def display_parameters(self):
        self.framed_log("PARAMETERS SUMMARY")
        for key, value in self.opt.items():
            print(f"{colored(key, 'blue')}: {value}")
            self.logger.info(f"{key}: {value}")

    def display_info(self):
        self.framed_log("INFO")
        for key, value in self.info.items():
            print(f"{colored(key, 'blue')}: {value}")
            self.logger.info(f"{key}: {value}")

    def log_one_train_epoch(self, epoch_info):
        for key, value in epoch_info.items():
            if isinstance(value, float):
                value = round(value, 4)
            print(f"{colored(key, 'blue')}[train]: {value}")
            self.logger.info(f"{key}[train]: {value}")

    def log_one_test_epoch(self, epoch_info):
        for key, value in epoch_info.items():
            if isinstance(value, float):
                value = round(value, 4)
            print(f"{colored(key, 'blue')}[eval]: {value}")
            self.logger.info(f"{key}[eval]: {value}")

    def framed_log(self, message):
        line = "=" * len(message)
        self.logger.info(line)
        self.logger.info(message)
        self.logger.info(line)
        print(line)
        print(message)
        print(line)

    def build_memory(self):
        self.memory = ThresholdReplayMemory(self.opt.buffer_size)
        self.info["memory"] = self.memory

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

    def save_model(self, filename="weigths.pth"):
        """Save weights in log folder."""
        path = os.path.join(os.path.join(self.opt.logs, self.opt.training_id, filename))
        torch.save(self.policy_net.state_dict(), path)
