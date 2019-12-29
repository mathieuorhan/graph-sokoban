from itertools import count
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import gym_sokoban

from data.embedding import MinimalEmbedding, NoWallsEmbedding
from data.replay import ReplayMemory
from model.network import Net
from rl.explore import epsilon_greedy
from rl.optimize import optimize_model


class TargetQLearningTrainer:
    def __init__(
        self,
        optimizer,
        policy_net,
        target_net,
        gamma,
        env,
        eps,
        embedding,
        target_update,
        memory,
        batch_size,
        max_steps,
    ):
        self.embedding = embedding
        self.env = env
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.batch_size = batch_size
        self.max_steps = max_steps

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()

            state = self.embedding(self.env.render(self.embedding.RENDER_MODE))
            # The level might be unsolvable and we wouldn't know
            counter = range(self.max_steps) if self.max_steps else count()
            for t in counter:
                # Select and perform an action
                action = epsilon_greedy(state, self.policy_net, self.eps)
                _, reward, done, _ = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if done:
                    next_state = None
                else:
                    next_state = self.embedding(
                        self.env.render(self.embedding.RENDER_MODE)
                    )

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimize_model(
                    policy_net=self.policy_net,
                    optimizer=self.optimizer,
                    memory=self.memory,
                    gamma=self.gamma,
                    batch_size=self.batch_size,
                )

                if done:
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def eval(self, num_episodes):
        # Evaluate
        mean_cum_reward = 0
        solved_score = 0

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()

            # Difficulty : when to stop ?
            # Idea: stop if the exact same position happen twice
            position_hash = dict()
            cum_reward = 0

            pixels = self.env.render(self.embedding.RENDER_MODE)
            pixels_hash = hashlib.md5(pixels).hexdigest()
            position_hash[pixels_hash] = position_hash.get(pixels_hash, 0) + 1
            state = self.embedding(pixels)

            counter = count()
            for t in counter:
                # Select and perform an action
                action = epsilon_greedy(state, self.policy_net, eps=0)
                _, reward, done, _ = self.env.step(action)
                cum_reward += reward
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if done:
                    next_state = None
                else:
                    next_state = self.embedding(
                        self.env.render(self.embedding.RENDER_MODE)
                    )

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if done:
                    solved_score += 1

        mean_cum_reward /= num_episodes
        solved_score /= num_episodes
        print(f"Evuation done on {num_episodes} episodes.")
        print(f"Mean c. rewards={mean_cum_reward}, solving score={solved_score}")

    def optimize_model(self):
        pass
