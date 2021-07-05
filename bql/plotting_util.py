import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards(bql_avg_rewards_data_1, bql_avg_rewards_means_1, bql_avg_rewards_stds_1,
                 q_avg_rewards_data_1, q_avg_rewards_means_1, q_avg_rewards_stds_1,
                 ylim_rew,
                 file_name, markevery=10, optimal_steps = 10, optimal_reward = 95, sample_step = 1,
                 plot_opt=False):
    """
    Plots rewards, flags % and steps of two different configurations
    """
    #matplotlib.style.use("seaborn")
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    plt.rcParams.update({'font.size': 12})

    # ylims = (0, 920)

    # Plot BQL
    ax.plot(np.array(list(range(len(bql_avg_rewards_means_1[::sample_step])))) * sample_step,
            bql_avg_rewards_means_1[::sample_step], label=r"BQL", marker="s", ls='-', color="#599ad3",
            markevery=markevery)
    ax.fill_between(np.array(list(range(len(bql_avg_rewards_means_1[::sample_step])))) * sample_step,
                    bql_avg_rewards_means_1[::sample_step] - bql_avg_rewards_stds_1[::sample_step], bql_avg_rewards_means_1[::sample_step]
                    + bql_avg_rewards_stds_1[::sample_step],
                    alpha=0.35, color="#599ad3")

    # Plot QL
    ax.plot(np.array(list(range(len(q_avg_rewards_means_1[::sample_step])))) * sample_step,
            q_avg_rewards_means_1[::sample_step], label=r"Q-Learning", marker="s", ls='-', color="r",
            markevery=markevery)
    ax.fill_between(np.array(list(range(len(q_avg_rewards_means_1[::sample_step])))) * sample_step,
                    q_avg_rewards_means_1[::sample_step] - q_avg_rewards_stds_1[::sample_step],
                    q_avg_rewards_means_1[::sample_step]
                    + q_avg_rewards_stds_1[::sample_step],
                    alpha=0.35, color="r")

    if plot_opt:
        ax.plot(np.array(list(range(len(bql_avg_rewards_means_1)))),
                [optimal_reward] * len(bql_avg_rewards_means_1), label=r"upper bound $\pi^{*}$",
                color="black",
                linestyle="dashed")

    ax.set_title(r"Episodic Rewards")
    ax.set_xlabel("\# Iteration", fontsize=20)
    ax.set_ylabel("Avg Episode Reward", fontsize=20)
    ax.set_xlim(0, len(bql_avg_rewards_means_1[::sample_step]) * sample_step)
    ax.set_ylim(ylim_rew[0], ylim_rew[1])
    #ax.set_ylim(ylim_rew)

    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=5, fancybox=True, shadow=True)
    #ax.legend(loc="lower right")
    ax.xaxis.label.set_size(13.5)
    ax.yaxis.label.set_size(13.5)

    fig.tight_layout()
    #plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.close(fig)

def read_csv():
    pass

if __name__ == '__main__':
    df_bql_0 = pd.read_csv("/home/kim/workspace/idsgame/bql/results/data/0/1625488542.0295575_train.csv")
    df_bql_299 = pd.read_csv("/home/kim/workspace/idsgame/bql/results/data/999/1625489956.2966099_train.csv")
    df_bql_999 = pd.read_csv("/home/kim/workspace/idsgame/bql/results/data/299/1625490890.822419_train.csv")
    bql_dfs = [df_bql_0, df_bql_299, df_bql_999]
    avg_bql_rewards_data = list(map(lambda df: df["hack_probability"].values, bql_dfs))
    avg_bql_rewards_means = np.mean(tuple(avg_bql_rewards_data), axis=0)
    avg_bql_rewards_stds = np.std(tuple(avg_bql_rewards_data), axis=0, ddof=1)

    df_q_0 = pd.read_csv("/home/kim/workspace/idsgame/tabular_q/results/data/0/1625491300.7774715_train.csv")
    df_q_299 = pd.read_csv("/home/kim/workspace/idsgame/tabular_q/results/data/999/1625491462.834143_train.csv")
    df_q_999 = pd.read_csv("/home/kim/workspace/idsgame/tabular_q/results/data/299/1625491618.800865_train.csv")
    q_dfs = [df_q_0, df_q_299, df_q_999]

    avg_q_rewards_data = list(map(lambda df: df["hack_probability"].values, q_dfs))
    avg_q_rewards_means = np.mean(tuple(avg_q_rewards_data), axis=0)
    avg_q_rewards_stds = np.std(tuple(avg_q_rewards_data), axis=0, ddof=1)

    ylim_rew = (0, 1)
    plot_rewards(avg_bql_rewards_data, avg_bql_rewards_means, avg_bql_rewards_stds,
                 avg_q_rewards_data, avg_q_rewards_means, avg_q_rewards_stds,
                 ylim_rew,
                 "bql_vs_q_rewards", markevery=5, optimal_steps=10, optimal_reward=1, sample_step=1,
                 plot_opt=True)
    # ylim_rew = (0.0, 1)
    # print(avg_bql_rewards_means)
    # plot_rewards(avg_bql_rewards_data, avg_bql_rewards_means, avg_bql_rewards_stds,
    #              avg_bql_rewards_data, avg_bql_rewards_means, avg_bql_rewards_stds,
    #              ylim_rew,
    #              "bql_vs_q_rewards", markevery=5, optimal_steps=10, optimal_reward=1, sample_step=1,
    #              plot_opt=True)




