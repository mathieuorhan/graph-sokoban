import random

from torch_geometric.nn import GraphUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Batch

from data.constants import Transition
from data.dataset import InMemorySokobanDataset
from data.graph_env import GraphEnv
from model.network import Net
from rl.abstract_trainer import AbstractTrainer
from rl.explore import epsilon_greedy_only_graph, best_from_nodes
from rl.schedulers import AnnealingScheduler


class QLearningTrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.random_generator = random.Random()

    def build_scheduler(self):
        self.scheduler = AnnealingScheduler(
            self.opt.eps_max, self.opt.eps_min, self.opt.eps_stop_step
        )

    def build_env(self):
        self.env = GraphEnv()

    def build_datasets(self):
        self.dataset_train = InMemorySokobanDataset(
            self.opt.train_path, self.embedding, device=self.device
        )
        self.dataset_test = InMemorySokobanDataset(
            self.opt.train_path, self.embedding, device=self.device
        )

    def build_networks(self):
        self.policy_net = GraphUNet(
            in_channels=self.embedding.NUM_NODES_FEATURES,
            hidden_channels=self.opt.unet_hidden_channels,
            out_channels=1,
            depth=self.opt.unet_depth,
            pool_ratios=self.opt.unet_pool_ratios,
            sum_res=self.opt.unet_sum_res,
            act=self.opt.unet_act,
        ).to(self.device)

        self.target_net = GraphUNet(
            in_channels=self.embedding.NUM_NODES_FEATURES,
            hidden_channels=self.opt.unet_hidden_channels,
            out_channels=1,
            depth=self.opt.unet_depth,
            pool_ratios=self.opt.unet_pool_ratios,
            sum_res=self.opt.unet_sum_res,
            act=self.opt.unet_act,
        ).to(self.device)

        self.target_net.eval()

    def train_one_epoch(self):
        print(f"=== EPOCH {self.epoch} [eps={self.scheduler.epsilon:.2f}] ===")
        self.policy_net.train()

        # Init counters
        epoch_info = {}
        epoch_info["mean_cum_reward"] = 0
        epoch_info["solved"] = 0
        epoch_info["deadlocks"] = 0

        # Sample the episodes
        ep_indexes = list(range(len(self.dataset_train)))
        random.shuffle(ep_indexes)

        for ep_idx in ep_indexes:
            # Generate one episode
            ep_info = self.train_one_episode(ep_idx)

            epoch_info["mean_cum_reward"] += ep_info["cum_reward"] / len(
                self.dataset_train
            )
            epoch_info["deadlocks"] += ep_info["deadlocks"]
            epoch_info["solved"] += ep_info["solved"]

        self.log_one_train_epoch(epoch_info)
        return epoch_info

    def log_one_train_epoch(self, epoch_info):
        reward = epoch_info["mean_cum_reward"]
        solved = epoch_info["solved"]
        deadlocks = epoch_info["deadlocks"]
        train_size = len(self.dataset_train)
        print(
            f"[Train] Mean reward: {reward:.2f}. Solved: {solved}/{train_size}. Deadlocks: {deadlocks}/{train_size}"
        )

    def log_one_test_epoch(self, epoch_info):
        reward = epoch_info["mean_cum_reward"]
        solved = epoch_info["solved"]
        # deadlocks = epoch_info["deadlocks"]
        test_size = len(self.dataset_test)
        print(
            f"[Eval.] Mean reward: {reward:.2f}. Solved: {solved}/{test_size}."  # Deadlocks: {deadlocks}/{train_size}"
        )

    def train_one_episode(self, episode_idx):
        ep_info = {}
        ep_info["cum_reward"] = 0.0
        ep_info["solved"] = 0
        ep_info["deadlocks"] = 0
        # Initialize the environment and state
        self.env.reset(self.dataset_train[episode_idx])

        for t in range(self.opt.max_steps):
            state = self.env.render()
            # Select and perform an action
            with torch.no_grad():
                action_node = epsilon_greedy_only_graph(
                    state,
                    self.policy_net,
                    self.scheduler.epsilon,
                    self.random_generator,
                    self.opt.walls_prob,
                    self.opt.static_prob,
                )
                next_state, reward, done, info = self.env.step(action_node)
            if info["deadlock"] and not self.opt.no_penalize_deadlocks:
                reward += self.opt.reward_deadlocks
            ep_info["cum_reward"] += reward
            reward = torch.tensor(reward, device=self.device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action_node, next_state, reward)

            # Perform one step of the optimization (on the target network)
            self.optimize_model()

            if done:
                ep_info["solved"] = 1
                break

            if info["deadlock"]:
                ep_info["deadlocks"] += 1
                if self.opt.early_stop_deadlocks:
                    break
                if self.opt.go_back_after_deadlocks:
                    # Go back directly to previous state
                    self.env.state = state
        return ep_info

    def optimize_model(self):
        # Sample a batch from buffer if available
        if len(self.memory) < self.opt.batch_size:
            return
        batch = self.memory.sample(self.opt.batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # in a single graph using pytorch_geometric Batch class
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        if any(non_final_mask):
            non_final_next_states = Batch.from_data_list(
                [s for s in batch.next_state if s is not None]
            ).to(self.device)

        state_batch = Batch.from_data_list(batch.state)
        # state_batch = state_batch.to(device)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = torch.zeros(self.opt.batch_size, device=self.device)
        state_action_values_batch = self.policy_net(
            state_batch.x, state_batch.edge_index
        )

        # TODO Try to remove for loop
        for i in range(self.opt.batch_size):
            state_action_values[i] = state_action_values_batch[state_batch.batch == i][
                action_batch[i]
            ]

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = torch.zeros(
                (self.opt.batch_size, 1), device=self.device
            )
            target_prediction = self.target_net(
                non_final_next_states.x, non_final_next_states.edge_index
            )
            neighbor_mask = non_final_next_states.mask.squeeze()
            if any(non_final_mask):
                next_state_values[non_final_mask], _ = scatter_max(
                    target_prediction[neighbor_mask],
                    non_final_next_states.batch[neighbor_mask],
                    dim=0,
                )
            next_state_values = next_state_values.squeeze()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.opt.gamma
        ) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def eval_one_epoch(self):
        self.policy_net.eval()

        # Init counters
        epoch_info = {}
        epoch_info["mean_cum_reward"] = 0
        epoch_info["solved"] = 0
        epoch_info["deadlocks"] = 0

        # Sample the episodes
        ep_indexes = list(range(len(self.dataset_test)))

        for ep_idx in ep_indexes:
            # Generate one episode
            ep_info = self.eval_one_episode(ep_idx)

            epoch_info["mean_cum_reward"] += ep_info["cum_reward"] / len(
                self.dataset_test
            )
            epoch_info["deadlocks"] += ep_info["deadlocks"]
            epoch_info["solved"] += ep_info["solved"]

        self.log_one_test_epoch(epoch_info)
        return epoch_info

    def eval_one_episode(self, episode_idx):
        ep_info = {}
        ep_info["cum_reward"] = 0.0
        ep_info["solved"] = 0
        ep_info["deadlocks"] = 0
        # Initialize the environment and state
        self.env.reset(self.dataset_test[episode_idx])
        for t in range(self.opt.max_steps_eval):
            state = self.env.render()
            # Select and perform an action
            with torch.no_grad():
                scores = self.policy_net(state.x, state.edge_index)
                action_node = best_from_nodes(
                    scores, state, self.opt.walls_prob == 0, self.opt.static_prob == 0,
                )
                next_state, reward, done, info = self.env.step(action_node)
            # Observe new state
            if done:
                ep_info["solved"] += 1
                break
            else:
                ep_info["cum_reward"] += reward

        return ep_info

