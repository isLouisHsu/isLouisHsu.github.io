import os
import json
import copy
import time
import random
from typing import *
from tqdm import trange
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class ActorNet(nn.Module):

    def __init__(self, input_size: int, num_actions: int, feature_size: int = 128) -> None:
        super(ActorNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        h_out = w_out = input_size // 4
        conv_output_size = feature_size * h_out * w_out
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

        self.num_actions = num_actions

    def forward(self, state: torch.Tensor, action: torch.Tensor = None):
        x = self.feature_extractor(state)
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)                  # (batch_size, num_actions)
        proba = F.softmax(logits, dim=-1)           # (batch_size, num_actions)

        if action is None:
            return proba, None
        
        # 在这里计算logproba
        log_proba = F.log_softmax(logits, dim=-1)   # (batch_size, num_actions)
        log_proba_selected = log_proba.gather(1, action.long().unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        return proba, log_proba_selected


class CriticNet(nn.Module):

    def __init__(self, input_size: int, feature_size: int = 128) -> None:
        super(CriticNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        h_out = w_out = input_size // 4
        conv_output_size = feature_size * h_out * w_out
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor):
        # 输入张量形状应该是 1 x input_size x input_size (C x H x W)
        x = self.feature_extractor(state)
        x = x.view(x.size(0), -1)  # 将多维特征图展平为一维向量
        return self.fc_layers(x)


class Utils():

    @staticmethod
    def set_seed(seed: int) -> None:
        """设置 Python 环境的所有常用随机数生成器的种子。"""
        if seed is None: 
            return
        random.seed(seed)  # Python's built-in random module
        np.random.seed(seed)  # Numpy library
        os.environ['PYTHONHASHSEED'] = str(seed)  # Environment variable

        # TensorFlow 2.x
        # import tensorflow as tf
        # tf.random.set_seed(seed)

        # PyTorch - If you are using PyTorch, you would also need to set its seed
        import torch
        torch.manual_seed(seed)
        # if you are using CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # Other libraries might also have their own random number generators.

    @staticmethod
    def whiten_sequence(sequence: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
        # 如果总元素数 <= 1，std 必然为 0，直接处理
        if sequence.numel() <= 1:
            return sequence - sequence.mean() if shift_mean else sequence.clone()
        mean, std = sequence.mean(), sequence.std()
        # 避免全零方差导致爆炸
        if std.item() < 1e-8:
            return sequence - mean if shift_mean else sequence.clone()
        whiten = (sequence - mean) / (std + 1e-8)
        if not shift_mean:
            whiten += mean
        return whiten


class DataUtils():
    
    @staticmethod
    def get_env(size: int = 8, is_slippery: bool = True, render_mode: str = None) -> gym.Env:
        return gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=size),
            is_slippery=is_slippery,
            render_mode=render_mode,
        )

    @staticmethod
    def build_static_grid(env: gym.Env) -> torch.Tensor:
        """ 从 env.unwrapped.desc 构造静态网格通道（S/G/F/H -> 0/1/2/3） """
        mapping = {b'S': 0.0, b'G': 1.0, b'F': 2.0, b'H': 3.0}
        desc = env.unwrapped.desc  # np.ndarray of bytes, shape (H, W)
        H, W = desc.shape
        grid = torch.empty((H, W), dtype=torch.float32)
        for i in range(H):
            for j in range(W):
                grid[i, j] = mapping[desc[i, j]]
        return grid

    @staticmethod
    def make_state_tensor(static_grid: torch.Tensor, obs: int) -> torch.Tensor:
        """ 根据 obs（离散索引）构造位置 one-hot 通道，并与静态网格通道堆叠 """
        H, W = static_grid.shape
        pos = torch.zeros((H, W), dtype=torch.float32)
        pos[obs // W, obs % W] = 1.0
        return torch.stack([static_grid, pos], dim=0)  # (2, H, W)

    @staticmethod
    @torch.no_grad()
    def sample_action(actor_model: nn.Module, state: torch.Tensor) -> Tuple[int, float]:
        device = next(actor_model.parameters()).device
        state = state.unsqueeze(0).float().to(device)  # (1, 2, H, W)
        probas, _ = actor_model(state)
        dist = torch.distributions.Categorical(probas)
        action = dist.sample()
        action_log_proba = dist.log_prob(action)
        return int(action.item()), float(action_log_proba.item())

    @staticmethod
    @torch.no_grad()
    def sample_round(env: gym.Env, actor_model: nn.Module, render_mode: str = None) -> List[Dict[str, Any]]:
        sequence = []
        score = None
        obs, info = env.reset()

        static_grid = DataUtils.build_static_grid(env)
        state = DataUtils.make_state_tensor(static_grid, obs)

        while True:
            if render_mode in ("rgb_array", "human"):
                env.render()
                time.sleep(0.3)
            action, _ = DataUtils.sample_action(actor_model, state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = DataUtils.make_state_tensor(static_grid, obs)

            sequence.append((state, action))
            state = next_state

            if terminated or truncated:
                sequence.append((state, None))
                score = float(reward)
                break

        states, actions = list(zip(*sequence))

        return dict(states=list(states), actions=list(actions), score=score)

    @staticmethod 
    @torch.no_grad()
    def sample_batch(actor_model: nn.Module, batch_size: int, group_size: int, **env_args) -> List[Dict[str, Any]]:
        actor_model.eval()
        examples = []
        for i in range(batch_size):
            env = DataUtils.get_env(**env_args)
            try:
                for i in range(group_size):
                    examples.append(DataUtils.sample_round(env, actor_model))
            finally:
                env.close()
        return examples


@dataclass
class Config():

    version: str = "v0"
    seed: int = 42
    frozen_lake_size: int = 4
    frozen_lake_slippery: bool = False
    num_actions: int = 4

    whiten_rewards: bool = False

    max_steps: int = 1000
    save_steps: int = 100
    batch_size: int = 32
    group_size: int = 8
    num_updates_per_batch: int = 1
    max_grad_norm: float = 0.5

    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = None

    def __post_init__(self):
        self.output_dir = os.path.join("./", self.version)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving to {self.output_dir}")


class Inferer():

    def __init__(self, config: Config, step_no: int, render_mode: str = "human") -> None:
        self.config = config
        self.step_no = step_no
        self.render_mode = render_mode

        # 读取模型
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_no:06d}")
        print(f"Loading model states from {save_dir}")
        self.actor_model = ActorNet(self.config.frozen_lake_size, self.config.num_actions).to(self.config.device)
        self.actor_model.load_state_dict(torch.load(os.path.join(save_dir, "actor.pt")))
        self.actor_model.eval()

    @torch.no_grad()
    def infer(self, ) -> None:
        # 初始化环境
        env = DataUtils.get_env(
            self.config.frozen_lake_size,
            self.config.frozen_lake_slippery,
            render_mode=self.render_mode,
        )
        return DataUtils.sample_round(env, self.actor_model, render_mode=self.render_mode)


class Trainer():

    def __init__(self, config: Config) -> None:
        self.config = config
        self.writer = SummaryWriter(
            os.path.join(config.output_dir, "logs/")
        )

    @torch.no_grad()
    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def update_model(self) -> None:
        raise NotImplementedError
    
    def save_model(self, step_no: int) -> str:
        raise NotImplementedError
    
    def train(self):
        for step_no in trange(self.config.max_steps, desc="Training...", total=self.config.max_steps):
            # 采样一批数据
            batch = DataUtils.sample_batch(
                self.actor_model, 
                self.config.batch_size, 
                self.config.group_size, 
                size=self.config.frozen_lake_size, 
                is_slippery=self.config.frozen_lake_slippery,
            )
            # 准备输入
            batch = self.prepare_inputs(batch)
            # 更新模型参数
            metrics = self.update_model(batch)
            # 打印参数
            print(json.dumps(metrics, ensure_ascii=False))
            for score_name, score_value in metrics.items():
                self.writer.add_scalar(score_name, score_value, step_no)  
            # 保存模型
            if step_no > 0 and step_no % self.config.save_steps == 0:
                model_path = self.save_model(step_no)
                print(f"Step [{step_no+1}/{self.config.max_steps}] model saved at {model_path}")


@dataclass
class PPOConfig(Config):

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4

    gamma: float = 0.9
    lam: float = 0.95

    critic_loss_coef: float = 0.5


class PPOTrainer(Trainer):

    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)

        self.actor_model = ActorNet(config.frozen_lake_size, config.num_actions).to(config.device)
        self.critic_model = CriticNet(config.frozen_lake_size).to(config.device)
        self.reference_model = None         # 预训练模型作为reference模型，但该实验无预训练模型

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=config.critic_learning_rate)
    
    def compute_gae(self, step_level_rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        sequence_length = step_level_rewards.size(0)
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(sequence_length)):  # 优势函数依赖于未来的值，所以从终点往回推
            next_value = values[t + 1] if t + 1 < sequence_length else 0.0              # 最后一个时间步，没有后续状态了，相当于假设 episode 结束，价值为 0
            delta = step_level_rewards[t] + self.config.gamma * next_value - values[t]  # 计算TD误差（Temporal Difference Error）：
                                                                                        #   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam       # 递归计算 GAE 优势值：
                                                                                        #   \A^{GAE}(s_t, a_t) = \delta_t + \gamma \lambda \delta_{t+1} + (\gamma \lambda) ** 2 \delta_{t+2} + ...
                                                                                        # 当 λ = 1，接近蒙特卡洛优势（即多步，高方差低偏差）；
                                                                                        # 当 λ = 0，退化为单步TD误差（即单步 \delta_t，低方差高偏差）；
                                                                                        # 取中间值，平衡偏差与方差。
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=-1)                    # (sequence_length,)
        return advantages
        
    @torch.no_grad()
    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch = copy.deepcopy(batch)
        for example_no in range(self.config.batch_size * self.config.group_size):
            example = batch[example_no]
            states: List[str] = example["states"]                                       # (sequence_length,)
            actions: List[int] = example["actions"]                                     # (sequence_length,)
            score: float = example["score"]
            sequence_length: int = len(states)

            # step 1. 计算每一步时，采取动作的对数概率（action_log_probs）和状态价值（values）
            encode_states = torch.stack(states, dim=0).float()                          # (sequence_length, channel, height, width)
            encode_actions = torch.tensor(actions[:-1], dtype=torch.int64)              # (sequence_length - 1,)
            _, action_log_probs = self.actor_model(encode_states[:-1], encode_actions)  # (sequence_length - 1,)
            values = self.critic_model(encode_states).squeeze(-1)                       # (sequence_length,)

            # step 2. 计算步级奖励（step_level_rewards），如果有参考模型（reference_model），这里应该：
            # 1）计算step-level的KL散度作为步级奖励；
            # 2）把序列级奖励加到最后一步
            step_level_rewards = [0.0] * (sequence_length - 1) + [score]
            step_level_rewards = torch.tensor(step_level_rewards, dtype=torch.float32)  # (sequence_length,) [0.0, 0.0, 0.0, ..., 1.0]
            if self.config.whiten_rewards:
                step_level_rewards = Utils.whiten_sequence(step_level_rewards, shift_mean=False)

            # step 3. GAE（Generalized Advantage Estimation），计算每一步的优势值（advantages）和回报（returns）
            advantages = self.compute_gae(step_level_rewards, values)                   # (sequence_length,)
            returns = advantages + values                                               # 计算回报值，作为critic model的groundtruth
                                                                                        # 已知：
            advantages = Utils.whiten_sequence(advantages)

            example["action_log_probs"] = action_log_probs                              # (sequence_length - 1,)
            example["values"] = values                                                  # (sequence_length,)
            example["advantages"] = advantages                                          # (sequence_length,)
            example["returns"] = returns                                                # (sequence_length,)
        
        return batch

    def update_model(self, batch: List[Dict[str, Any]]) -> None:
        self.actor_model.train()
        log_actor_loss = 0.0
        log_critic_loss = 0.0
        # 更新模型参数
        for epoch_no in range(self.config.num_updates_per_batch):
            # 使用“步数加权”的累计器
            device = next(self.actor_model.parameters()).device
            total_actor_loss = torch.tensor(0.0, device=device)
            total_actor_steps = 0   # 记录步数，防止序列长度影响样本权重
            total_critic_loss = torch.tensor(0.0, device=device)
            total_critic_steps = 0  # 记录步数，防止序列长度影响样本权重

            for example_no in range(self.config.batch_size * self.config.group_size):
                example = batch[example_no]
                states: List[str] = example["states"]                                       # (sequence_length,)
                actions: List[int] = example["actions"]                                     # (sequence_length,)
                old_action_log_probs: torch.Tensor = example["action_log_probs"]            # (sequence_length - 1,)
                advantages: torch.Tensor = example["advantages"]                            # (sequence_length,)
                returns: torch.Tensor = example["returns"]                                  # (sequence_length,)

                # 重新前向
                encode_states = torch.stack(states, dim=0).float()                          # (sequence_length, channel, height, width)
                encode_actions = torch.tensor(actions[:-1], dtype=torch.int64)              # (sequence_length - 1,)
                probas, action_log_probs = self.actor_model(encode_states[:-1], encode_actions)  # (sequence_length - 1,)
                values = self.critic_model(encode_states).squeeze(-1)                       # (sequence_length,)

                # actor：逐步损失，不做 mean
                ratio = torch.exp(action_log_probs - old_action_log_probs)                  # (sequence_length - 1,)
                step_actor_loss = - torch.min(
                    ratio * advantages[:-1],
                    torch.clamp(
                        ratio, 
                        1 - self.config.clip_epsilon, 
                        1 + self.config.clip_epsilon,
                    ) * advantages[:-1]
                )                                                                           # (sequence_length - 1,)

                # 熵奖励，最大化行动熵以鼓励探索
                entropy = - (probas * torch.log(torch.clamp(probas, min=1e-8))).sum(dim=1)                                  # (sequence_length,)
                step_actor_loss = step_actor_loss - self.config.entropy_coef * entropy

                # critic：逐步 MSE，不做 mean
                step_critic_loss = 0.5 * torch.square(values - returns)                     # (sequence_length,)

                # 累加总和与有效步数
                total_actor_loss += step_actor_loss.sum()
                total_actor_steps += step_actor_loss.numel()

                total_critic_loss += (self.config.critic_loss_coef * step_critic_loss).sum()
                total_critic_steps += step_critic_loss.numel()

                # 如需记录每个样本的指标（仅用于日志，不用于梯度）
                example["actor_loss"] = step_actor_loss.mean().item()
                example["critic_loss"] = step_critic_loss.mean().item()

            # 用“总和 / 总步数”得到 batch 级损失，确保每个时间步权重一致
            actor_loss = total_actor_loss / max(1, total_actor_steps)
            critic_loss = total_critic_loss / max(1, total_critic_steps)
            log_actor_loss += (actor_loss.item() / self.config.num_updates_per_batch)
            log_critic_loss += (critic_loss.item() / self.config.num_updates_per_batch)

            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            # 更新critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

        # 打印指标（保持不变）
        metrics = {
            "score/mean": torch.tensor([e["score"] for e in batch]).mean().item(),
            "score/max": torch.tensor([e["score"] for e in batch]).max().item(),
            "score/min": torch.tensor([e["score"] for e in batch]).min().item(),
            "actor_loss": log_actor_loss,
            "critic_loss": log_critic_loss,
        }
        return metrics

    def save_model(self, step_no: int) -> str:
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_no:06d}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor_model.state_dict(), os.path.join(save_dir, f"actor.pt"))
        torch.save(self.critic_model.state_dict(), os.path.join(save_dir, f"critic.pt"))
        return save_dir

@dataclass
class GRPOConfig(Config):

    actor_learning_rate: float = 1e-4


class GRPOTrainer(Trainer):

    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)

        self.actor_model = ActorNet(config.frozen_lake_size, config.num_actions).to(config.device)
        self.reference_model = None         # 预训练模型作为reference模型，但该实验无预训练模型

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=config.actor_learning_rate)

    def compute_grpo(self, rewards: torch.Tensor) -> torch.Tensor:
        # 如果总元素数 <= 1，std 必然为 0，直接处理
        if rewards.numel() <= 1:
            return rewards - rewards.mean()
        mean, std = rewards.mean(), rewards.std()
        # 避免全零方差导致爆炸
        if std.item() < 1e-8:
            return rewards - mean
        return (rewards - mean) / (std + 1e-8)
        
    @torch.no_grad()
    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch = copy.deepcopy(batch)
        device = self.config.device

        for group_no in range(self.config.batch_size):
            group_start = group_no * self.config.group_size
            group_end = group_start + self.config.group_size
            group = batch[group_start: group_end]

            # GRPO（Group Relative Policy Optimization）: group relative advantage estimation
            grouped_rewards = torch.tensor([example["score"] for example in group]).float().to(device)
            grouped_advantages = self.compute_grpo(grouped_rewards)     # len = group_size

            for example_no in range(self.config.group_size):
                example = group[example_no]
                states: List[str] = example["states"]                                       # (sequence_length,)
                actions: List[int] = example["actions"]                                     # (sequence_length,)
                score: float = example["score"]
                sequence_length: int = len(states)

                # step 1. 计算每一步时，采取动作的对数概率（action_log_probs）和状态价值（values）
                encode_states = torch.stack(states, dim=0).float()                          # (sequence_length, channel, height, width)
                encode_actions = torch.tensor(actions[:-1], dtype=torch.int64)              # (sequence_length - 1,)
                _, action_log_probs = self.actor_model(encode_states[:-1], encode_actions)  # (sequence_length - 1,)

                # step 2. GRPO（Group Relative Policy Optimization）: group relative advantage estimation
                # DeepSeek原文：Outcome supervision provides the normalized reward at the end of each output 𝑜𝑖 and 
                #              sets the advantages 𝐴ˆ𝑖,𝑡 of all tokens in the output as the normalized reward
                advantages = grouped_advantages[example_no]                     # (1,)

                example["action_log_probs"] = action_log_probs.detach()         # (sequence_length,)
                example["advantages"] = advantages.detach()                     # (sequence_length,)
        
        return batch

    def update_model(self, batch: List[Dict[str, Any]]) -> None:
        self.actor_model.train()
        log_actor_loss = 0.0
        # 更新模型参数
        for epoch_no in range(self.config.num_updates_per_batch):
            # 使用“步数加权”的累计器
            device = next(self.actor_model.parameters()).device
            total_actor_loss = torch.tensor(0.0, device=device)
            total_actor_steps = 0   # 记录步数，防止序列长度影响样本权重

            for example_no in range(self.config.batch_size * self.config.group_size):
                example = batch[example_no]
                states: List[str] = example["states"]                                       # (sequence_length,)
                actions: List[int] = example["actions"]                                     # (sequence_length,)
                old_action_log_probs: torch.Tensor = example["action_log_probs"]            # (sequence_length - 1,)
                advantages: torch.Tensor = example["advantages"]                            # (1,)

                # 重新前向
                encode_states = torch.stack(states, dim=0).float()                          # (sequence_length, channel, height, width)
                encode_actions = torch.tensor(actions[:-1], dtype=torch.int64)              # (sequence_length - 1,)
                probas, action_log_probs = self.actor_model(encode_states[:-1], encode_actions)  # (sequence_length - 1,)

                # actor：逐步损失，不做 mean
                ratio = torch.exp(action_log_probs - old_action_log_probs)                  # (sequence_length - 1,)
                step_actor_loss = - torch.min(
                    ratio * advantages,
                    torch.clamp(
                        ratio, 
                        1 - self.config.clip_epsilon, 
                        1 + self.config.clip_epsilon,
                    ) * advantages
                )                                                                           # (sequence_length - 1,)

                # 熵奖励，最大化行动熵以鼓励探索
                entropy = - (probas * torch.log(torch.clamp(probas, min=1e-8))).sum(dim=1)                                  # (sequence_length,)
                step_actor_loss = step_actor_loss - self.config.entropy_coef * entropy

                # 累加总和与有效步数
                total_actor_loss += step_actor_loss.sum()
                total_actor_steps += step_actor_loss.numel()

                # 如需记录每个样本的指标（仅用于日志，不用于梯度）
                example["actor_loss"] = step_actor_loss.mean().item()

            # 用“总和 / 总步数”得到 batch 级损失，确保每个时间步权重一致
            actor_loss = total_actor_loss / max(1, total_actor_steps)
            log_actor_loss += (actor_loss.item() / self.config.num_updates_per_batch)

            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

        # 打印指标（保持不变）
        metrics = {
            "score/mean": torch.tensor([e["score"] for e in batch]).mean().item(),
            "score/max": torch.tensor([e["score"] for e in batch]).max().item(),
            "score/min": torch.tensor([e["score"] for e in batch]).min().item(),
            "actor_loss": log_actor_loss,
        }
        return metrics

    def save_model(self, step_no: int) -> str:
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{step_no:06d}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor_model.state_dict(), os.path.join(save_dir, f"actor.pt"))
        return save_dir


if __name__ == "__main__":
    parser = ArgumentParser(description="""
# 最简单的实现，没有进行异步采样、训练
# 为方便理解，没有采取全向量化的计算方式，比如回报（returns）的计算，也没有用到GPU加速

# # 环境说明：https://gymnasium.farama.org/environments/toy_text/frozen_lake/
# desc = generate_random_map(size=8)
# env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)

# RL运算参考：
# PPO：https://github.com/huggingface/trl/blob/20cc58d7772ae660792c7b5249d8b817986a547d/trl/trainer/ppo_trainer.py#L448
# GRPO：https://github.com/huggingface/trl/blob/9e5e60c9334d0d6d52498da4de68632148fceafb/trl/trainer/grpo_trainer.py#L1362
    """)
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--observation_size", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--frozen_lake_size", type=int, default=4)

    parser.add_argument("--adv_estimator", type=str, choices=["ppo", "grpo"], default="ppo")
    parser.add_argument("--max_steps", type=int, default=1000, help="总的训练步数")
    parser.add_argument("--save_steps", type=int, default=100, help="每隔若干步数保存一次模型")
    parser.add_argument("--batch_size", type=int, default=32, help="每个step中的样本数量")
    parser.add_argument("--group_size", type=int, default=8,  help="每个样本采样的个数，每个step中的总样本数是(batch_size * group_size)")
    parser.add_argument("--num_updates_per_batch", type=int, default=1, help="每个采样的批次用于迭代模型的轮数")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-4, help="actor模型学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=3e-4, help="critic模型学习率")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--whiten_rewards", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)

    parser.add_argument("--entropy_coef", type=float, default=0.0, help="熵奖励系数，用于最大化行动熵以鼓励探索")
    parser.add_argument("--critic_loss_coef", type=float, default=1.0, help="critic模型的权重系数")

    args = parser.parse_args()

    Utils.set_seed(args.seed)

    if args.adv_estimator == "ppo":
        ppo_config = PPOConfig(
            version=args.version,
            seed=args.seed,
            frozen_lake_size=args.frozen_lake_size,
            num_actions=args.num_actions,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            group_size=args.group_size,
            num_updates_per_batch=args.num_updates_per_batch,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            max_grad_norm=args.max_grad_norm,
            whiten_rewards=args.whiten_rewards,
            gamma=args.gamma,
            lam=args.lam,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            critic_loss_coef=args.critic_loss_coef,
        )
        # inferer = Inferer(ppo_config, step_no=900)
        # for i in range(100):
        #     inferer.infer()
        # exit(0)

        trainer = PPOTrainer(ppo_config)
        trainer.train()

    elif args.adv_estimator == "grpo":
        grpo_config = GRPOConfig(
            version=args.version,
            seed=args.seed,
            frozen_lake_size=args.frozen_lake_size,
            num_actions=args.num_actions,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            batch_size=args.batch_size,
            group_size=args.group_size,
            num_updates_per_batch=args.num_updates_per_batch,
            actor_learning_rate=args.actor_learning_rate,
            max_grad_norm=args.max_grad_norm,
            whiten_rewards=args.whiten_rewards,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
        )

        trainer = GRPOTrainer(grpo_config)
        trainer.train()

