import random
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Batch

from data.constants import Transition
from data.dataset import InMemorySokobanDataset
import data.utils as utils
from model.graph_centered import GraphCenteredNet
from rl.abstract_trainer import AbstractTrainer
from rl.explore import epsilon_greedy_gc
from rl.schedulers import AnnealingScheduler
from rl.qlearning_trainer import QLearningTrainer
from data.graph_env import GraphEnv


class QLearningGraphCenteredTrainer(QLearningTrainer):
    def build_env(self):
        self.env = GraphEnv(
            embedding=self.embedding, device=self.device, stop_if_unreachable=False
        )

    def build_networks(self):
        self.policy_net = GraphCenteredNet(self.embedding.NUM_NODES_FEATURES, None)
        self.target_net = GraphCenteredNet(self.embedding.NUM_NODES_FEATURES, None)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

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
                action = epsilon_greedy_gc(
                    state, self.policy_net, self.scheduler.epsilon, device=self.device
                )
                action_node = utils.direction_to_node_idx(state, action)
                next_state, reward, done, info = self.env.step(action_node)

            ep_info["cum_reward"] += reward
            reward = torch.tensor(reward, device=self.device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            self.optimize_model()
            self.scheduler.step()

            if done:
                ep_info["solved"] = 1
                break

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
            )

        state_batch = Batch.from_data_list(batch.state)
        # cuda, (batch_size, 1)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        scores, _, _ = self.policy_net(
            x=state_batch.x,
            edge_index=state_batch.edge_index,
            edge_attr=state_batch.edge_attr,
            u=None,
            batch=state_batch.batch.to(self.device),
        )
        # (batch_size, 1)
        state_action_values = scores.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = torch.zeros(self.opt.batch_size, device=self.device)
            if any(non_final_mask):
                target_scores, _, _ = self.target_net(
                    x=non_final_next_states.x,
                    edge_index=non_final_next_states.edge_index,
                    edge_attr=non_final_next_states.edge_attr,
                    u=None,
                    batch=non_final_next_states.batch.to(self.device),
                )
                next_state_values[non_final_mask] = target_scores.max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.opt.gamma
        ) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        print(loss, reward_batch.mean())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if not self.opt.no_clamp_gradient:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1

    def eval_one_episode(self, episode_idx):
        with torch.no_grad():
            ep_info = {}
            ep_info["cum_reward"] = 0.0
            ep_info["solved"] = 0
            ep_info["deadlocks"] = 0
            # Initialize the environment and state
            self.env.reset(self.dataset_test[episode_idx])
            for t in range(self.opt.max_steps_eval):
                state = self.env.render()
                # Select and perform an action
                action = epsilon_greedy_gc(
                    state, self.policy_net, 0, device=self.device
                )
                action_node = utils.direction_to_node_idx(state, action)
                next_state, reward, done, info = self.env.step(action_node)
                # Observe new state
                if done:
                    ep_info["solved"] += 1
                    break
                else:
                    ep_info["cum_reward"] += reward

            return ep_info

    def render_one_episode(self, episode_idx):
        # TODO
        return

