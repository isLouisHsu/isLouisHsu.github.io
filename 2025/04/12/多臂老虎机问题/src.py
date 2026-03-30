import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.style as style

# 设置一个更美观的绘图风格
style.use('seaborn-v0_8-whitegrid')

class BernoulliBandit:
    def __init__(self, n_arms, seed=None):
        self.rng = np.random.default_rng(seed)
        # 每个臂的真实成功概率随机生成
        self.p = self.rng.uniform(0.0, 1.0, size=n_arms)
        self.n_arms = n_arms

    def pull(self, arm):
        return 1 if self.rng.random() < self.p[arm] else 0

def ucb1(n_arms, steps, bandit, alpha=3.0, seed=None, record_ucb=True):
    """
    UCB1: 选择 argmax_a [ mean[a] + sqrt(alpha * ln(t) / (2 * n[a])) ]
    alpha 常用 3.0，也可调节探索强度
    """
    rng = np.random.default_rng(seed)

    counts = np.zeros(n_arms, dtype=int)
    means = np.zeros(n_arms, dtype=float)
    actions = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps, dtype=float)

    if record_ucb:
        mean_hist = np.zeros((steps, n_arms), dtype=float)
        delta_hist = np.zeros((steps, n_arms), dtype=float)
        ucb_hist = np.zeros((steps, n_arms), dtype=float)
    else:
        mean_hist = delta_hist = ucb_hist = None

    # 先保证每个臂至少拉一次
    t = 0
    for a in range(n_arms):
        if t >= steps:
            break
        r = bandit.pull(a)
        counts[a] += 1
        means[a] = r
        actions[t] = a
        rewards[t] = r

        if record_ucb:
            # 在早期，为了数值稳定性，分母可以加一个极小值或特殊处理
            # 这里为了简化，直接用 counts，因为我们保证了它不为 0
            # 使用 t+1 避免 log(0)
            delta = np.sqrt(alpha * np.log(t + 1) / (2 * counts))
            ucb = means + delta
            mean_hist[t] = means
            delta_hist[t] = delta
            ucb_hist[t] = ucb
        t += 1

    # 主循环
    for t in range(t, steps):
        # 计算每个 arm 的 UCB 值
        # np.log(t) 而不是 t+1，因为此时 t 代表已经过去的步数，从 n_arms 开始
        delta = np.sqrt(alpha * np.log(t) / (2 * counts))
        ucb = means + delta

        if record_ucb:
            mean_hist[t] = means
            delta_hist[t] = delta
            ucb_hist[t] = ucb

        # 选择 UCB 值最大的 arm
        a = int(np.argmax(ucb))
        r = bandit.pull(a)

        # 增量更新均值
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]

        actions[t] = a
        rewards[t] = r

    return actions, rewards, counts, means, mean_hist, delta_hist, ucb_hist


def thompson_sampling(n_arms, steps, bandit, seed=None, record_posteriors=True):
    """
    Thompson Sampling: 为每个臂维护一个 Beta 分布。
    """
    rng = np.random.default_rng(seed)

    # beta_params 存储每个臂的 Beta 分布参数 [alpha, beta]
    # 初始为 Beta(1, 1)，即均匀分布（无信息先验）
    beta_params = np.ones((n_arms, 2), dtype=float)
    
    actions = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps, dtype=float)

    # 用于记录每一步的后验分布参数（可选）
    if record_posteriors:
        posterior_hist = np.zeros((steps, n_arms, 2), dtype=float)
    else:
        posterior_hist = None

    for t in range(steps):
        # 1. 采样：从每个臂的当前 Beta 后验分布中抽取一个样本
        samples = rng.beta(beta_params[:, 0], beta_params[:, 1])

        # 记录（在选臂之前记录这一时刻的后验分布）
        if record_posteriors:
            # .copy() 很重要，否则 history 会全部指向最后的状态
            posterior_hist[t] = beta_params.copy()

        # 2. 选择：选择样本值最大的臂
        a = int(np.argmax(samples))
        
        # 3. 拉动选择的臂并观察奖励
        r = bandit.pull(a)

        # 4. 更新后验分布：
        # 如果 r=1 (成功), alpha 增加 1
        # 如果 r=0 (失败), beta 增加 1
        beta_params[a, 0] += r
        beta_params[a, 1] += (1 - r)

        # 记录动作和奖励
        actions[t] = a
        rewards[t] = r
        
    # 计算最终的alpha和beta，可以用来推断每个臂的成功次数和失败次数
    # 成功次数 = alpha - 1, 失败次数 = beta - 1
    counts = (beta_params[:, 0] + beta_params[:, 1] - 2).astype(int)
    # 贝叶斯估计的均值是 alpha / (alpha + beta)
    estimated_means = beta_params[:, 0] / (beta_params[:, 0] + beta_params[:, 1])

    return actions, rewards, counts, estimated_means, posterior_hist


def plot_ucb1_results(bandit, actions, rewards, counts, mean_hist, ucb_hist):
    """
    功能强大的可视化函数，为每个臂单独绘制 UCB 演化图。
    """
    T = len(rewards)
    n_arms = bandit.n_arms
    true_p = bandit.p
    best_arm = np.argmax(true_p)
    time_steps = np.arange(1, T + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, n_arms))

    # --- 计算子图网格布局 ---
    # 总图数 = 臂的数量 + 2 (悔憾图和次数图)
    n_plots = n_arms + 2
    n_cols = 2
    # 向上取整计算行数
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    # 将 2D 数组扁平化为 1D，方便索引
    axes = axes.flatten()
    fig.suptitle('UCB1 Algorithm Analysis (Individual Arm View)', fontsize=20)

    # --- 图 1 to n_arms: 每个臂的 UCB 值与均值估计的演化 ---
    for a in range(n_arms):
        ax = axes[a]
        # 绘制真实的概率 p (水平线)
        ax.axhline(y=true_p[a], color='gray', ls=':', lw=2,
                   label=f'True p={true_p[a]:.3f}')
        # 绘制均值估计
        ax.plot(time_steps, mean_hist[:, a], color=colors[a], ls='-',
                label='Estimated Mean')
        # 用半透明区域填充均值和 UCB 之间的空间，代表“不确定性”
        ax.fill_between(time_steps, mean_hist[:, a], ucb_hist[:, a],
                         color=colors[a], alpha=0.3, label='Uncertainty (UCB-Mean)')

        title = f'Arm {a} Analysis'
        if a == best_arm:
            title += ' (Best Arm)'
            ax.set_facecolor('gold') # 给最优臂的子图一个背景色
            ax.patch.set_alpha(0.15)
            
        ax.set_title(title)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 倒数第二个图: 累积伪悔憾 ---
    ax_regret = axes[n_arms]
    best_p = true_p.max()
    instant_regret = best_p - true_p[actions]
    cumulative_regret = np.cumsum(instant_regret)

    ax_regret.plot(time_steps, cumulative_regret, color='crimson')
    ax_regret.set_title('Cumulative Pseudo-Regret Over Time')
    ax_regret.set_xlabel('Time Steps')
    ax_regret.set_ylabel('Cumulative Regret')
    ax_regret.text(T * 0.05, cumulative_regret[-1] * 0.8,
                   f'Total Regret: {cumulative_regret[-1]:.2f}', fontsize=12)

    # --- 最后一个图: 各臂被选择次数 ---
    ax_counts = axes[n_arms + 1]
    arm_indices = np.arange(n_arms)
    bar_colors = [colors[i] for i in arm_indices]
    
    bars = ax_counts.bar(arm_indices, counts, color=bar_colors, edgecolor='black')
    ax_counts.set_title('Total Pull Counts per Arm')
    ax_counts.set_xlabel('Arm Index')
    ax_counts.set_ylabel('Number of Pulls')
    ax_counts.set_xticks(arm_indices)
    ax_counts.bar_label(bars, label_type='edge')
    # 高亮最优臂
    bars[best_arm].set_color('gold')
    bars[best_arm].set_edgecolor('black')
    ax_counts.legend([bars[best_arm]], [f'Best Arm ({best_arm})'])

    # --- 隐藏多余的子图 ---
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("source/_drafts/多臂老虎机问题/ucb1.png")
    plt.show()


def plot_thompson_sampling_results(bandit, actions, rewards, counts, posterior_hist, title=None):
    """
    Visualizes the results of the Thompson Sampling algorithm.

    - For each arm: plots the evolution of the Beta posterior distribution at different time steps.
    - Plots the cumulative pseudo-regret over time.
    - Plots the total pull counts for each arm.
    
    Args:
        bandit (BernoulliBandit): The bandit environment.
        actions (np.array): The sequence of actions taken.
        rewards (np.array): The sequence of rewards received.
        counts (np.array): The final pull counts for each arm.
        posterior_hist (np.array): History of posterior parameters (alpha, beta) for each arm.
        title (str, optional): The main title for the plot.
    """
    T = len(rewards)
    n_arms = bandit.n_arms
    true_p = bandit.p
    best_arm = np.argmax(true_p)
    time_steps = np.arange(1, T + 1)
    
    # Consistent colors for arms across different plots
    arm_colors = plt.cm.viridis(np.linspace(0, 1, n_arms))

    # --- Setup subplot grid ---
    n_plots = n_arms + 2
    n_cols = 2 if n_arms > 1 else 1 # Handle single-arm case
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    # Set main title
    fig.suptitle(title or 'Thompson Sampling Analysis', fontsize=20)

    # --- Plot 1 to n_arms: Posterior Distribution Evolution ---
    # Select a few time steps (logarithmically spaced) to show the evolution
    if T > 10:
        t_points = np.logspace(1, np.log10(T - 1), num=4, dtype=int)
    else: # Handle small T
        t_points = np.linspace(0, T - 1, num=min(T, 4), dtype=int)
    
    # Use a sequential colormap where darker colors represent later times
    time_colors = plt.cm.cividis_r(np.linspace(0.2, 1, len(t_points)))
    x_pdf = np.linspace(0, 1, 300)

    for a in range(n_arms):
        ax = axes[a]
        # Plot the true probability as a vertical line
        ax.axvline(true_p[a], color='black', ls='--', lw=2, label=f'True p={true_p[a]:.3f}')
        
        # Plot the PDF at different time steps
        for i, t in enumerate(t_points):
            alpha, beta = posterior_hist[t, a]
            if alpha > 0 and beta > 0: # Ensure valid Beta parameters
                pdf = stats.beta.pdf(x_pdf, alpha, beta)
                ax.plot(x_pdf, pdf, color=time_colors[i], label=f't={t+1}')

        plot_title = f'Arm {a}: Posterior Distribution'
        if a == best_arm:
            plot_title += ' (Best Arm)'
            ax.set_facecolor('gold')
            ax.patch.set_alpha(0.15)

        ax.set_title(plot_title)
        ax.set_xlabel('Success Probability p')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot n_arms + 1: Cumulative Pseudo-Regret ---
    ax_regret = axes[n_arms]
    best_p = true_p.max()
    instant_regret = best_p - true_p[actions]
    cumulative_regret = np.cumsum(instant_regret)

    ax_regret.plot(time_steps, cumulative_regret, color='crimson')
    ax_regret.set_title('Cumulative Pseudo-Regret Over Time')
    ax_regret.set_xlabel('Time Steps')
    ax_regret.set_ylabel('Cumulative Regret')
    ax_regret.text(T * 0.05, cumulative_regret[-1] * 0.8,
                   f'Total Regret: {cumulative_regret[-1]:.2f}', fontsize=12)
    ax_regret.grid(True)

    # --- Plot n_arms + 2: Total Pull Counts ---
    ax_counts = axes[n_arms + 1]
    arm_indices = np.arange(n_arms)
    
    bars = ax_counts.bar(arm_indices, counts, color=arm_colors, edgecolor='black')
    ax_counts.set_title('Total Pull Counts per Arm')
    ax_counts.set_xlabel('Arm Index')
    ax_counts.set_ylabel('Number of Pulls')
    ax_counts.set_xticks(arm_indices)
    ax_counts.bar_label(bars, label_type='edge')
    
    # Highlight the best arm
    bars[best_arm].set_color('gold')
    bars[best_arm].set_edgecolor('black')
    ax_counts.legend([bars[best_arm]], [f'Best Arm ({best_arm})'])

    # --- Hide unused subplots ---
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("source/_drafts/多臂老虎机问题/ts.png")
    plt.show()


if __name__ == "__main__":
    N = 4
    T = 5000
    bandit = BernoulliBandit(n_arms=N, seed=0)

    actions, rewards, counts, means, mean_hist, delta_hist, ucb_hist = ucb1(
        n_arms=N, steps=T, bandit=bandit, alpha=3.0, seed=0, record_ucb=True
    )

    best_arm = int(np.argmax(bandit.p))
    best_p = bandit.p[best_arm]
    total_reward = rewards.sum()
    regret = T * best_p - total_reward

    print("True p per arm:", np.round(bandit.p, 3))
    print("Best arm:", best_arm, "best p:", round(best_p, 3))
    print("Pull counts:", counts)
    print("Estimated means:", np.round(means, 3))
    print("Total reward:", int(total_reward))
    print("Pseudo-regret (from true p):", round(regret, 2))

    # 调用新的、按臂分开的可视化函数
    plot_ucb1_results(bandit, actions, rewards, counts, mean_hist, ucb_hist)


    actions, rewards, counts, estimated_means, posterior_hist = thompson_sampling(
        n_arms=N, steps=T, bandit=bandit, seed=0, record_posteriors=True
    )

    best_arm = int(np.argmax(bandit.p))
    best_p = bandit.p[best_arm]
    total_reward = rewards.sum()
    regret = T * best_p - total_reward

    print("Algorithm: Thompson Sampling")
    print("True p per arm:", np.round(bandit.p, 3))
    print("Best arm:", best_arm, "best p:", round(best_p, 3))
    print("Total pull counts:", counts)
    print("Estimated means (from posteriors):", np.round(estimated_means, 3))
    print("Total reward:", int(total_reward))
    print("Pseudo-regret:", round(regret, 2))

    # 可视化 Beta 后验分布随时间的变化
    plot_thompson_sampling_results(
        bandit=bandit,
        actions=actions,
        rewards=rewards,
        counts=counts,
        posterior_hist=posterior_hist,
        title="Thompson Sampling Analysis"
    )

