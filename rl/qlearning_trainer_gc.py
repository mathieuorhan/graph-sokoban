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
from data.graph_env import GraphEnv
from data.prio_replay import PrioritisedReplayBuffer
import data.utils as utils
from model.graph_centered import (
    GraphCenteredNet,
    SimpleGraphCenteredNet,
    GraphCenteredNetV2,
)
from rl.abstract_trainer import AbstractTrainer
from rl.explore import epsilon_greedy_gc
from rl.schedulers import AnnealingScheduler
from rl.qlearning_trainer import QLearningTrainer


class QLearningGraphCenteredTrainer(QLearningTrainer):
    def build_env(self):
        self.env = GraphEnv(
            embedding=self.embedding, device=self.device, stop_if_unreachable=False
        )

    def build_networks(self):
        self.policy_net = GraphCenteredNetV2(
            self.embedding.NUM_NODES_FEATURES,
            None,
            num_message_passing=self.opt.num_message_passing,
            # ratio=1.0,
            hiddens=self.opt.hiddens,
            aggr="max",
            device=self.device,
        )
        self.target_net = GraphCenteredNetV2(
            self.embedding.NUM_NODES_FEATURES,
            None,
            num_message_passing=self.opt.num_message_passing,
            # ratio=1.0,
            hiddens=self.opt.hiddens,
            aggr="max",
            device=self.device,
        )
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.info["model"] = self.policy_net

    def train_one_episode(self, episode_idx):
        ep_info = {}
        ep_info["cum_reward"] = 0.0
        ep_info["solved"] = 0
        ep_info["deadlocks"] = 0
        ep_info["loss"] = 0.0
        # Initialize the environment and state
        self.env.reset(self.dataset_train[episode_idx])

        for t in range(self.opt.max_steps):
            state = self.env.render()
            # Select and perform an action
            with torch.no_grad():
                action = epsilon_greedy_gc(
                    state,
                    self.policy_net,
                    self.scheduler.epsilon,
                    device=self.device,
                    opt=self.opt,
                )
                action_node = utils.direction_to_node_idx(state, action)
                next_state, reward, done, info = self.env.step(action_node)

            ep_info["cum_reward"] += reward

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            loss = self.optimize_model()
            self.scheduler.step()

            ep_info["loss"] += loss

            if done:
                ep_info["solved"] = 1
                break

        return ep_info

    def optimize_model(self):
        # Sample a batch from buffer if available
        if len(self.memory) < self.opt.batch_size:
            return 0.0
        batch = self.memory.sample(self.opt.batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # in a single graph using pytorch_geometric Batch class
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
        )

        if any(non_final_mask):
            non_final_mask = non_final_mask.to(self.device)
            non_final_next_states = Batch.from_data_list(
                [s for s in batch.next_state if s is not None]
            ).to(self.device)

        state_batch = Batch.from_data_list(batch.state).to(self.device)

        # cuda, (batch_size, 1)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        scores, _, _ = self.policy_net(
            x=state_batch.x,
            edge_index=state_batch.edge_index,
            edge_attr=state_batch.edge_attr,
            u=None,
            batch=state_batch.batch,
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
                    batch=non_final_next_states.batch,
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
        # print(loss.item(), reward_batch.mean().item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if not self.opt.no_clamp_gradient:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1

        return loss.item()

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
                    state, self.policy_net, 0, device=self.device, opt=self.opt
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

    def save_model(self, filename="weigths.pth"):
        """Save weights in log folder."""
        path = os.path.join(os.path.join(self.opt.logs, self.opt.training_id, filename))
        torch.save(self.policy_net.state_dict(), path)


class QLearningPrioritizedBufferGraphCenteredTrainer(QLearningGraphCenteredTrainer):
    # TODO : factorize

    def build_memory(self):
        self.memory = PrioritisedReplayBuffer(self.opt, device=self.device)

    def optimize_model(self):
        # Sample a batch from buffer if available
        if len(self.memory) < self.opt.batch_size:
            return (0.0, 0.0)
        batch, importance_sampling_weights = self.memory.sample(self.opt.batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # in a single graph using pytorch_geometric Batch class
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
        )

        if any(non_final_mask):
            non_final_mask = non_final_mask.to(self.device)
            non_final_next_states = Batch.from_data_list(
                [s for s in batch.next_state if s is not None]
            ).to(self.device)

        state_batch = Batch.from_data_list(batch.state).to(self.device)

        # cuda, (batch_size, 1)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        scores, _, _ = self.policy_net(
            x=state_batch.x,
            edge_index=state_batch.edge_index,
            edge_attr=state_batch.edge_attr,
            u=None,
            batch=state_batch.batch,
        )

        # (batch_size, 1)
        state_action_values = scores.gather(1, action_batch).squeeze()

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
                    batch=non_final_next_states.batch,
                )
                next_state_values[non_final_mask] = target_scores.max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.opt.gamma
        ) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values, reduction="none",
        )
        loss = loss * importance_sampling_weights
        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if not self.opt.no_clamp_gradient:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        td_errors = (
            state_action_values.data.cpu().numpy()
            - expected_state_action_values.data.cpu().numpy()
        )
        self.memory.update_td_errors(td_errors)
        mean_td_error = td_errors.mean()

        self.update_count += 1
        return loss.item(), mean_td_error.item()

    def train_one_episode(self, episode_idx):
        ep_info = {}
        ep_info["cum_reward"] = 0.0
        ep_info["solved"] = 0
        ep_info["td_error"] = 0
        ep_info["loss"] = 0.0
        # Initialize the environment and state
        self.env.reset(self.dataset_train[episode_idx])

        for t in range(self.opt.max_steps):
            state = self.env.render()
            # Select and perform an action
            with torch.no_grad():
                action = epsilon_greedy_gc(
                    state,
                    self.policy_net,
                    self.scheduler.epsilon,
                    device=self.device,
                    opt=self.opt,
                )
                action_node = utils.direction_to_node_idx(state, action)
                next_state, reward, done, info = self.env.step(action_node)

            ep_info["cum_reward"] += reward

            # Observe new state
            if done:
                next_state = None

            # Perform one step of the optimization (on the target network)
            loss, mean_td_error = self.optimize_model()

            # Store the transition in memory
            max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
            self.memory.add_experience(
                max_td_error_in_experiences, state, action, reward, next_state
            )
            self.scheduler.step()

            ep_info["loss"] += loss
            ep_info["td_error"] += mean_td_error

            if done:
                ep_info["solved"] = 1
                break

        return ep_info
