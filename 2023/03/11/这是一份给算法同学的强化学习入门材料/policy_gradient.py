import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym

mode = "train"
# mode = "test"
LearningRate = 0.01
Gamma = 0.9             # Gamma越大越容易收敛
env = gym.make('CartPole-v1')
env = env.unwrapped
state_number = env.observation_space.shape[0]
action_number = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''policygrandient第一步先建网络'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.in_to_y1 = nn.Linear(state_number,20)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2 = nn.Linear(20,10)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(10,action_number)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,input_state):
        input_state = self.in_to_y1(input_state)
        input_state = F.relu(input_state)
        input_state = self.y1_to_y2(input_state)
        input_state = torch.sigmoid(input_state)
        act = self.out(input_state)
        return F.softmax(act,dim=-1)

class PG():

    def __init__(self):
        self.policy = Net().to(device)
        self.rewards, self.obs, self.acts = [],[],[]
        self.renderflag = False
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LearningRate)

    '''第二步 定义选择动作函数'''
    def choose(self, input_state):
        input_state = torch.FloatTensor(input_state).to(device)
        action_probas = self.policy(input_state)
        action = Categorical(action_probas).sample().item()
        return action

    '''第三步 存储每一个回合的数据'''
    def store_transtion(self, s, a, r):
        self.obs.append(s)
        self.acts.append(a)
        self.rewards.append(r)

    '''第四步 学习'''
    def learn(self):
        self.optimizer.zero_grad()
        # 按照policy gradient推导的公式计算奖励
        # reward_tensor = torch.FloatTensor(np.array(self.rewards)).to(device).sum()
        # 计算时刻t到回合结束的奖励值的累加，并对奖励归一化，减去平均数再除以标准差
        running_add = 0
        discounted_ep_r = np.zeros_like(self.rewards)
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * Gamma + self.rewards[t]
            discounted_ep_r[t] = running_add            # 改进2：为每步t赋予不同权重
        discounted_ep_r -= np.mean(discounted_ep_r)     # 改进1：增加一个奖励基准$b$，这里用均值
        # 我们可以用G值直接进行学习，但一般来说，对数据进行归一化处理后，训练效果会更好
        discounted_ep_r /= np.std(discounted_ep_r)
        reward_tensor = torch.FloatTensor(discounted_ep_r).to(device)
        # 状态、动作
        state_tensor = torch.FloatTensor(np.array(self.obs)).to(device)
        action_tensor = torch.LongTensor(self.acts).to(device)
        log_prob = torch.log(self.policy(state_tensor))                     # log_prob是拥有两个动作概率的张量，一个左动作概率，一个右动作概率
        log_prob = log_prob[np.arange(len(action_tensor)), action_tensor]   # np.arange(len(action_tensor))是log_prob的索引，取出采取动作对应的对数概率
        # action_tensor由0、1组成，于是log_prob[np.arange(len(action_tensor)), action_tensor]就可以取到我们已经选择了的动作的概率，是拥有一个动作概率的张量
        loss = - (reward_tensor * log_prob).mean()
        loss.backward()
        self.optimizer.step()
        # 清空该回合记录
        self.obs, self.acts, self.rewards = [], [], []

'''训练'''
def train():
    print("训练PG中...")
    pg = PG()
    for i in range(1000):
        r = 0
        observation, _ = env.reset()
        while True:
            if pg.renderflag: 
                env.render()
            # 用策略网络选择动作
            action = pg.choose(observation)
            # 与环境产生交互
            observation_, reward, done, truncated,info = env.step(action)
            # 预处理，修改reward，你也可以不修改奖励，直接用reward，都能收敛
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r3 = 3 * r1 + r2
            r += reward
            pg.store_transtion(observation, action, r3)
            # 一回合结束，用该回合数据训练
            if done:
                pg.learn()
                break
            # 更新状态
            observation = observation_
        print("\rEp: {} rewards: {}".format(i, r), end="")
        if i % 10 == 0 and i > 100:
            save_data = {'net': pg.policy.state_dict(), 'opt': pg.optimizer.state_dict(), 'i': i}
            torch.save(save_data, "model_PG.pth")

def test():
    print("测试PG中...")
    pg = PG()
    checkpoint = torch.load("model_PG.pth")
    pg.policy.load_state_dict(checkpoint['net'])
    env = gym.make('CartPole-v1', render_mode="human")
    for j in range(10):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            state = torch.FloatTensor(state)
            # 用策略网络选择动作
            action = pg.choose(state)
            # 与环境产生交互
            new_state, reward, done, truncated, info = env.step(action)  # 执行动作
            total_rewards += reward
            if done:
                print("Score", total_rewards)
                break
            state = new_state
    env.close()

if __name__ == "__main__":
    train()
    test()