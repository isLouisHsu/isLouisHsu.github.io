import numpy as np
import pandas as pd
import time

np.random.seed(42)

N_STATES = 6                # 1维世界的宽度(-----T)
ACTIONS = ['left', 'right'] # 探索者的可用动作
EPSILON = 0.9               # 贪婪度 greedy
ALPHA = 0.1                 # 学习率
GAMMA = 0.9                 # 奖励递减值
MAX_EPISODES = 13           # 最大回合数
FRESH_TIME = 0.3            # 移动间隔时间


def build_q_table(n_states, actions):
    """ 新建Q表格，Q(s, a)表示在位置s处采取a行为的行为值 """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table 全 0 初始
        columns=actions,  # columns 对应的是行为名称
    )
    return table


# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""


# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    """ 以\epsilon-greedy策略，选择当前s处选择的动作a 
    
    以90%概率贪婪选择，10%概率随机选择
    """
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.any() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # 贪婪模式
    return action_name


def get_env_feedback(S, A):
    """ 在位置s处采取动作a，求取状态s'、奖励r """
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate：目前在s=4的位置，再向右移动1，到达s=5(T)
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall：已经到达最左端，不能再向左
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r[{} - {}] {}'.format(episode, step_counter, interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        S = 0  # 回合初始位置
        is_terminated = False  # 是否回合结束
        update_env(S, episode, step_counter)  # 环境更新
        while not is_terminated:
            
            # 根据Q表格选择状态s采取的动作a，并作用于环境得到反馈和奖励
            A = choose_action(S, q_table)  # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]  # 估算的(状态-行为)值
            
            # 计算下一个状态的所能拿到的最大奖励
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # terminate this episode

            # q_table 更新：用下一个状态的所能拿到的最大奖励，作为当前状态行为的目标值
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            step_counter += 1; S = S_  # 探索者移动到下一个 state
            update_env(S, episode, step_counter)    # 环境更新

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

