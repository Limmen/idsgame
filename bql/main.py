import os
import time
import glob
import sys
import gym
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util
from bql.bql_agent import BQLAgent
from bql.q_agent_config import QAgentConfig


def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir


def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    config_path = os.path.join(default_output_dir(), './config.json')
    return config_path


def setup_agent_attacker() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=50,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos", num_episodes=5001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs",
                                  eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data",
                                  a_mu0=-10, a_lambda0=4.0, a_alpha0=3.0, a_beta0=3.0, num_mixture_samples=1000)
    env_name = "idsgame-random_defense-v3"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir(),
                                 title="TrainingQAgent vs DefendMinimalDefender",
                                 run_many=True, random_seeds=[0, 999, 299])
    env = gym.make(client_config.env_name, idsgame_config=client_config.idsgame_config,
                   save_dir=client_config.output_dir + "/results/data/" + str(client_config.random_seed),
                   initial_state_path=client_config.initial_state_path)
    attacker = BQLAgent(env, client_config.q_agent_config)
    return attacker, client_config, env


def setup_agent_defender() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=50,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos", num_episodes=5001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs",
                                  eval_frequency=1000, attacker=False, defender=True, video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data",
                                  a_mu0=-10, a_lambda0=4.0, a_alpha0=3.0, a_beta0=3.0, num_mixture_samples=1000)
    # env_name = "idsgame-random_defense-v3"
    env_name = "idsgame-maximal_attack-v3"
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir(),
                                 title="TrainingQAgent vs DefendMinimalDefender",
                                 run_many=True, random_seeds=[0, 999, 299])
    env = gym.make(client_config.env_name, idsgame_config=client_config.idsgame_config,
                   save_dir=client_config.output_dir + "/results/data/" + str(client_config.random_seed),
                   initial_state_path=client_config.initial_state_path)
    defender = BQLAgent(env, client_config.q_agent_config)
    return defender, client_config, env

def train(train_agent, time_str, random_seed):
    train_agent.train()
    train_result = train_agent.train_result
    eval_result = train_agent.eval_result
    if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
        train_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_train" + ".csv"
        train_result.to_csv(train_csv_path)
        eval_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_eval" + ".csv"
        eval_result.to_csv(eval_csv_path)

def setup_train(config: ClientConfig, random_seed):
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir, random_seed)
    logger = util.setup_logger("tabular_q_vs_random_defense-v3", config.output_dir + "/results/logs/" +
                               str(random_seed) + "/",
                               time_str=time_str)
    config.q_agent_config.save_dir = default_output_dir() + "/results/data/" + str(random_seed) + "/"
    config.q_agent_config.video_dir = default_output_dir() + "/results/videos/" + str(random_seed) + "/"
    config.q_agent_config.gif_dir = default_output_dir() + "/results/gifs/" + str(random_seed) + "/"
    # config.q_agent_config.dqn_config.tensorboard_dir = default_output_dir() + "/results/tensorboard/" \
    #                                                    + str(random_seed) + "/"
    config.logger = logger
    config.q_agent_config.logger = logger
    config.q_agent_config.random_seed = random_seed
    config.random_seed = random_seed
    config.q_agent_config.to_csv(
        config.output_dir + "/results/hyperparameters/" + str(random_seed) + "/" + time_str + ".csv")
    return time_str

if __name__ == '__main__':

    # Attacker
    # train_agent, config, env = setup_agent_attacker()

    # Defender
    train_agent, config, env = setup_agent_defender()

    if not config.run_many:
        random_seed = 0
        time_str = setup_train(config, random_seed)
        train(train_agent, time_str, random_seed)
    else:
        for seed in config.random_seeds:
            # train_agent, config, env = setup_agent_attacker()
            train_agent, config, env = setup_agent_defender()
            time_str = setup_train(config, seed)
            train(train_agent, time_str, seed)

