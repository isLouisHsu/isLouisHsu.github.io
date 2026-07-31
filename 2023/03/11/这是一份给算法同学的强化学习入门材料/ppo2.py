import os
import random
import argparse
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum with PPO')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', 
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('Pendulum-v1', render_mode='human' if args.render else None).unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
torch.manual_seed(args.seed)
random.seed(args.seed)

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.tanh(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)  # 策略函数：输出分布（均值和标准差）


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO2():
    clip_epsilon = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32

    def __init__(self):
        super(PPO2, self).__init__()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()

    @torch.no_grad()
    def get_value(self, state):
        state = torch.from_numpy(state)
        value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'ppo2_actor_params.pkl')
        torch.save(self.critic_net.state_dict(), 'ppo2_critic_params.pkl')
    
    def load_param(self):
        self.actor_net.load_state_dict(torch.load('ppo2_actor_params.pkl'))
        self.critic_net.load_state_dict(torch.load('ppo2_critic_params.pkl'))

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        action_log_prob_old = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        del self.buffer[:]

        with torch.no_grad():
            reward = (reward + 8) / 8
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)
            # 动作价值函数 Q^{\pi}(s, a) = r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) V^{\pi}(s')
            target_v = reward + args.gamma * self.critic_net(next_state)
            # 优势函数 A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
            advantage = target_v - self.critic_net(state)

        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                # 行动策略 \pi(a|s;\tilde{\theta})
                mu, sigma = self.actor_net(state[index])
                dist = Normal(mu, sigma)
                action_log_prob = dist.log_prob(action[index])
                
                # # Actor-Critic(TD error)
                # action_loss = - (action_log_prob * advantage[index]).mean()

                # PPO2
                ratio = torch.exp(action_log_prob - action_log_prob_old[index]
                )   # 重要性采样系数 \frac{\pi(a|s;\tilde{\theta})}{\pi(a|s; \theta)}
                action_loss = - torch.min(
                    ratio * advantage[index],
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[index],
                ).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()


def main(is_training):
    agent = PPO2()
    
    if not is_training: 
        agent.load_param()
        args.render = True

    training_records = []
    running_reward = -1000

    for i_epoch in range(1000):
        score = 0
        state, _ = env.reset()
        if args.render: env.render()
        for t in range(200):
            # 评估策略 \pi(a|s;\theta)
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step([action])
            if args.render: env.render()

            if is_training:
                trans = Transition(state, action, action_log_prob, reward, next_state)  # s, a, \pi, r, s'
                if agent.store_transition(trans):
                    agent.update()

            score += reward
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainRecord(i_epoch, running_reward))
        if i_epoch % 10 == 0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break


if __name__ == '__main__':
    main(is_training=True)
    main(is_training=False)
