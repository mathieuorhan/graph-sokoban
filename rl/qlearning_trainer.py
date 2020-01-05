import random
import time
import os
import matplotlib.pyplot as plt
from termcolor import colored
from torch_geometric.nn import GraphUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Batch

from data.constants import Transition
from data.dataset import InMemorySokobanDataset
from data.utils import display_graph
from data.graph_env import GraphEnv
from model.network import Net, GATNet
from model.meta import MetaGNN
from rl.abstract_trainer import AbstractTrainer
from rl.explore import epsilon_greedy, best_from_nodes
from rl.schedulers import AnnealingScheduler


class QLearningTrainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)

    def build_scheduler(self):
        self.scheduler = AnnealingScheduler(
            self.opt.eps_max, self.opt.eps_min, self.opt.eps_stop_step
        )
        self.info["scheduler"] = AnnealingScheduler

    def build_env(self):
        self.env = GraphEnv(embedding=self.embedding, device=self.device)
        self.info["env"] = self.env

    def build_datasets(self):
        self.dataset_train = InMemorySokobanDataset(
            self.opt.train_path, self.embedding, device=self.device
        )
        self.dataset_test = InMemorySokobanDataset(
            self.opt.test_path, self.embedding, device=self.device
        )
        self.info["train_size"] = len(self.dataset_train)
        self.info["test_size"] = len(self.dataset_test)

    def build_networks(self):
        # self.policy_net = GraphUNet(
        #     in_channels=self.embedding.NUM_NODES_FEATURES,
        #     hidden_channels=self.opt.unet_hidden_channels,
        #     out_channels=1,
        #     depth=self.opt.unet_depth,
        #     pool_ratios=self.opt.unet_pool_ratios,
        #     sum_res=self.opt.unet_sum_res,
        #     act=self.opt.unet_act,
        # ).to(self.device)

        # self.target_net = GraphUNet(
        #     in_channels=self.embedding.NUM_NODES_FEATURES,
        #     hidden_channels=self.opt.unet_hidden_channels,
        #     out_channels=1,
        #     depth=self.opt.unet_depth,
        #     pool_ratios=self.opt.unet_pool_ratios,
        #     sum_res=self.opt.unet_sum_res,
        #     act=self.opt.unet_act,
        # ).to(self.device)

        # self.info["model"] = "GraphUNet"
        # self.policy_net = Net(nodes_features=self.embedding.NUM_NODES_FEATURES).to(
        #     self.device
        # )
        # self.target_net = Net(nodes_features=self.embedding.NUM_NODES_FEATURES).to(
        #     self.device
        # )
        n_nodes_features = self.embedding.NUM_NODES_FEATURES
        n_edges_features = self.embedding.NUM_EDGES_FEATURES
        hiddens = 64
        self.policy_net = MetaGNN(n_nodes_features, n_edges_features, hiddens).to(
            self.device
        )
        self.target_net = MetaGNN(n_nodes_features, n_edges_features, hiddens).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.info["model"] = self.policy_net

    def train_one_epoch(self):
        self.framed_log(f"EPOCH {self.epoch}")
        self.policy_net.train()

        # Init counters
        epoch_info = {}
        epoch_info["mean_cum_reward"] = 0.0
        epoch_info["solved"] = 0
        epoch_info["deadlocks"] = 0
        epoch_info["dataset size"] = len(self.dataset_train)
        epoch_info["epsilon"] = self.scheduler.epsilon
        epoch_info["time elapsed"] = time.time()
        epoch_info["mean_loss"] = 0.0

        # Sample the episodes
        ep_indexes = list(range(len(self.dataset_train)))
        random.shuffle(ep_indexes)

        for ep_idx in ep_indexes:
            # Generate one episode
            ep_info = self.train_one_episode(ep_idx)
            self.episodes_seen += 1

            epoch_info["mean_cum_reward"] += ep_info["cum_reward"] / len(
                self.dataset_train
            )
            epoch_info["deadlocks"] += ep_info["deadlocks"]
            epoch_info["solved"] += ep_info["solved"]
            epoch_info["mean_loss"] += ep_info["loss"]
            # Update the target network, copying all weights and biases in DQN
            if self.update_count % self.opt.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        epoch_info["mean_loss"] /= len(self.dataset_train)
        epoch_info["time elapsed"] = time.time() - epoch_info["time elapsed"]
        self.log_one_train_epoch(epoch_info)
        self.epoch += 1
        return epoch_info

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
                action_node = epsilon_greedy(
                    state, self.policy_net, self.scheduler.epsilon
                )
                next_state, reward, done, info = self.env.step(action_node)
            # if info["deadlock"] and not self.opt.no_penalize_deadlocks:
            #     reward += self.opt.reward_deadlocks
            ep_info["cum_reward"] += reward
            reward = torch.tensor(reward, device=self.device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, action_node, next_state, reward)

            # Perform one step of the optimization (on the target network)
            loss = self.optimize_model()
            self.scheduler.step()

            ep_info["loss"] += loss

            if done:
                ep_info["solved"] = 1
                break

            # if info["deadlock"]:
            #     ep_info["deadlocks"] += 1
            #     if self.opt.early_stop_deadlocks:
            #         break
            #     if self.opt.go_back_after_deadlocks:
            #         # Go back directly to previous state
            #         self.env.state = state
        return ep_info

    def optimize_model(self):
        # Sample a batch from buffer if available
        if len(self.memory) < self.opt.batch_size:
            return 0.0
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

        state_batch = Batch.from_data_list(batch.state)  # .to(self.device)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = torch.zeros(self.opt.batch_size, device=self.device)
        batch_idx = state_batch.batch.to(self.device)
        state_action_values_batch, _, _ = self.policy_net(
            x=state_batch.x,
            edge_index=state_batch.edge_index,
            edge_attr=state_batch.edge_attr,
            u=None,
            batch=batch_idx,
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
            if any(non_final_mask):
                non_final_batch_idx = non_final_next_states.batch.to(self.device)
                target_prediction, _, _ = self.target_net(
                    x=non_final_next_states.x,
                    edge_index=non_final_next_states.edge_index,
                    edge_attr=non_final_next_states.edge_attr,
                    u=None,
                    batch=non_final_batch_idx,
                )
                neighbor_mask = non_final_next_states.mask.squeeze()

                next_state_values[non_final_mask], _ = scatter_max(
                    target_prediction[neighbor_mask],
                    non_final_batch_idx[neighbor_mask],
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
        if not self.opt.no_clamp_gradient:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1

        return loss.item()

    def eval_one_epoch(self):
        self.policy_net.eval()

        # Init counters
        epoch_info = {}
        epoch_info["mean_cum_reward"] = 0
        epoch_info["solved"] = 0
        epoch_info["dataset size"] = len(self.dataset_test)
        # epoch_info["deadlocks"] = 0

        # Sample the episodes
        ep_indexes = list(range(len(self.dataset_test)))

        for ep_idx in ep_indexes:
            # Generate one episode
            ep_info = self.eval_one_episode(ep_idx)

            epoch_info["mean_cum_reward"] += ep_info["cum_reward"] / len(
                self.dataset_test
            )
            # epoch_info["deadlocks"] += ep_info["deadlocks"]
            epoch_info["solved"] += ep_info["solved"]

        self.log_one_test_epoch(epoch_info)
        return epoch_info

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
                scores, _, _ = self.policy_net(
                    x=state.x,
                    edge_index=state.edge_index,
                    edge_attr=state.edge_attr,
                    u=None,
                    batch=None,
                )
                action_node = best_from_nodes(scores, state)
                next_state, reward, done, info = self.env.step(action_node)
                # Observe new state
                if done:
                    ep_info["solved"] += 1
                    break
                else:
                    ep_info["cum_reward"] += reward

            return ep_info

    def render_one_episode(self, episode_idx):
        with torch.no_grad():
            # Initialize the environment and state
            self.env.reset(self.dataset_test[episode_idx])
            for t in range(self.opt.max_steps_eval):
                state = self.env.render()
                # Select and perform an action
                scores, _, _ = self.policy_net(
                    x=state.x,
                    edge_index=state.edge_index,
                    edge_attr=state.edge_attr,
                    u=None,
                    batch=None,
                )
                action_node = best_from_nodes(scores, state)

                next_state, reward, done, info = self.env.step(action_node)

                # Save the state display
                plt.figure()
                display_graph(state, scores)
                os.makedirs(
                    "./logs/{}/rendering".format(self.opt.training_id), exist_ok=True
                )
                os.makedirs(
                    os.path.join(
                        "./logs/{}/rendering".format(self.opt.training_id),
                        "epoch{}".format(self.epoch),
                    ),
                    exist_ok=True,
                )

                plt.savefig(
                    os.path.join(
                        "./logs/{}/rendering".format(self.opt.training_id),
                        "epoch{}/{}.png".format(self.epoch, t),
                    )
                )
                plt.close()

                if done:
                    break

    def save_model(self, filename="weigths.pth"):
        """Save weights in log folder."""
        path = os.path.join(os.path.join(self.opt.logs, self.opt.training_id, filename))
        torch.save(self.policy_net.state_dict(), path)
