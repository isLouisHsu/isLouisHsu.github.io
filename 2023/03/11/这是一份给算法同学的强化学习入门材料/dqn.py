import os
import gym
import numpy as np
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make('CartPole-v1')
env = env.unwrapped
state_number = env.observation_space.shape[0]
action_number = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_number, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_number),
        )

    def forward(self, state):
        q = self.layers(state)  # (batch_size, action_number)
        return q

class ExperienceReplayBuffer():
       
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    # 增加经验，因为经验数组是存放在deque中的，deque是双端队列，
    # 我们的deque指定了大小，当deque满了之后再add元素，则会自动把队首的元素出队
    def add(self,experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, continuous=False):
        # 防止越界
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = None
        if continuous:
            # 表示连续取batch_size个经验
            rand = np.random.randint(0, len(self.buffer) - batch_size)
            indices = list(range(rand, rand + batch_size))
        else:
            indices = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def clear(self):
        self.buffer.clear()


class DQN():

    def __init__(
        self, 
        epsilon=0.1, 
        epsilon_decrement=1e-6,
        memory_size=20000, 
        min_memory_size=200, 
        update_per_n_steps=5, 
        update_target_per_n_steps=200,
        batch_size=32, 
        gamma=0.99,
        alpha=1.0,
        lr=5e-4,
        weight_decay=0.0,
    ):
        self.epsilon = epsilon      # \epsilon-greedy
        self.epsilon_decrement = epsilon_decrement
        self.memory_size = memory_size
        self.min_memory_size = min_memory_size
        self.update_per_n_steps = update_per_n_steps
        self.update_target_per_n_steps = update_target_per_n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        
        self.buffer = ExperienceReplayBuffer(memory_size)
        self.model = Net()
        self.target_model = deepcopy(self.model)    # Fixed-Q-Target
        self.model.to(device); self.target_model.to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fct = nn.MSELoss()
    
    @torch.no_grad()
    def choose_action(self, state):
        """ \epsilon-greedy """
        action = None
        randval = np.random.random()    # [0.0, 1.0)
        if randval < self.epsilon:      # 随机选择
            action = np.random.randint(action_number)
        else:                           # 根据q选择
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q = self.model(state).squeeze(0)
            action = torch.argmax(q).item()

        # 动态更改e_greed,但不小于0.01
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decrement)
        return action

    def store_experience(self, experience):
        self.buffer.add(experience)
    
    def shoud_update(self, step):
        # 当经验回放数组中的经验数量足够多时（大于给定阈值，手动设定），每5个时间步训练一次
        return self.buffer.size() > self.min_memory_size and step % self.update_per_n_steps == 0

    @torch.no_grad()
    def update_target_model(self):
        state_dict = self.model.state_dict()
        for name, para in self.target_model.named_parameters():
            para.copy_(state_dict[name].data.clone() * self.alpha + para.data.clone() * (1. - self.alpha))
    
    def update(self, step):
        # Double DQN：每隔若干步，更新一次target
        if step % self.update_target_per_n_steps == 0:
            self.update_target_model()

        # 采样一批数据
        batch = self.buffer.sample(self.batch_size, continuous=False)
        get_tensor = lambda x: torch.tensor([b[x] for b in batch]).to(device)
        states = get_tensor(0).float()
        actions = get_tensor(1).long()
        rewards = get_tensor(2).float()
        next_states = get_tensor(3).float()
        done = get_tensor(4).long()

        # 计算target
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(dim=-1)[0]
            target = rewards + (1 - done) * self.gamma * max_next_q
        # 计算pred
        q = self.model(states)
        pred = torch.sum(q * F.one_hot(actions), dim=-1)
        # 计算损失，并更新model
        loss = self.loss_fct(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def train(agent, num_episodes=2000, render=False):
    step = 0
    for i in range(num_episodes):
        total_rewards = 0
        done = False
        state, _ = env.reset()
        while not done:
            step += 1
            if render: env.render()
            # 选择动作
            action = agent.choose_action(state)
            # 与环境产生交互
            next_state, reward, done, truncated, info = env.step(action)
            # 预处理，修改reward，你也可以不修改奖励，直接用reward，都能收敛
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r3 = 3 * r1 + r2
            # 经验回放
            agent.store_experience((state, action, r3, next_state, done))
            # 更新参数
            if agent.shoud_update(step):
                loss = agent.update(step)
            # 更新状态
            state = next_state
            total_rewards += reward

        if i % 50 == 0:
            print('episode:{} reward:{} epsilon:{} '.format(i, total_rewards, agent.epsilon))
        
def test(agent, num_episodes=10, render=False):
    env = gym.make('CartPole-v1', render_mode="human" if render else None)
    step = 0
    eval_rewards = []
    for i in range(num_episodes):
        total_rewards = 0
        done = False
        state, _ = env.reset()
        while not done:
            step += 1
            if render: env.render()
            # 选择动作
            action = agent.choose_action(state)
            # 与环境产生交互
            next_state, reward, done, truncated, info = env.step(action)
            # 更新状态
            state = next_state
            total_rewards += reward
        eval_rewards.append(total_rewards)
    return sum(eval_rewards) / len(eval_rewards)

if __name__ == "__main__":
    agent = DQN()
    train(agent, render=False)
    test(agent, render=True)
