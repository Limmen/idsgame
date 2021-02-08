import os
import sys
import gym
import time
from dqn_template.dqn_agent import DQNAgent
from dqn_template.dqn_config import DQNConfig
from dqn_template.client_config import ClientConfig
from dqn_template.q_agent_config import QAgentConfig
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.agent_type import AgentType
import dqn_template.util as util

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

def setup_agent():
    dqn_config = DQNConfig(input_dim=88, attacker_output_dim=80, hidden_dim=64, replay_memory_size=10000,
                           num_hidden_layers=1,
                           replay_start_size=1000, batch_size=32, target_network_update_freq=1000,
                           gpu=True, tensorboard=True, tensorboard_dir=default_output_dir() + "/results/tensorboard",
                           loss_fn="Huber", optimizer="Adam", lr_exp_decay=True, lr_decay_rate=0.9999)
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.00001, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos", num_episodes=20001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs",
                                  eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data", dqn_config=dqn_config,
                                  checkpoint_freq=5000)
    env_name = "idsgame-minimal_defense-v3"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.DQN_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir(),
                                 title="TrainingDQNAgent vs DefendMinimalDefender",
                                 run_many=True, random_seeds=[0, 999, 299, 399, 499])
    env = gym.make(client_config.env_name, idsgame_config=client_config.idsgame_config,
                   save_dir=client_config.output_dir + "/results/data/" + str(client_config.random_seed),
                   initial_state_path=client_config.initial_state_path)
    attacker = DQNAgent(env, client_config.q_agent_config)
    return attacker, client_config, env

def train(attacker, time_str):
    attacker.train()
    train_result = attacker.train_result
    eval_result = attacker.eval_result
    if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
        train_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_train" + ".csv"
        train_result.to_csv(train_csv_path)
        eval_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_eval" + ".csv"
        eval_result.to_csv(eval_csv_path)

def setup_train(config: ClientConfig, random_seed):
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir, random_seed)
    logger = util.setup_logger("dqn_vs_random_defense-v3", config.output_dir + "/results/logs/" +
                               str(random_seed) + "/",
                               time_str=time_str)
    config.q_agent_config.save_dir = default_output_dir() + "/results/data/" + str(random_seed) + "/"
    config.q_agent_config.video_dir = default_output_dir() + "/results/videos/" + str(random_seed) + "/"
    config.q_agent_config.gif_dir = default_output_dir() + "/results/gifs/" + str(random_seed) + "/"
    config.q_agent_config.dqn_config.tensorboard_dir = default_output_dir() + "/results/tensorboard/" \
                                                       + str(random_seed) + "/"
    config.logger = logger
    config.q_agent_config.logger = logger
    config.q_agent_config.random_seed = random_seed
    config.random_seed = random_seed
    config.q_agent_config.to_csv(
        config.output_dir + "/results/hyperparameters/" + str(random_seed) + "/" + time_str + ".csv")
    return time_str

if __name__ == '__main__':
    random_seed = 0
    attacker, config, env = setup_agent()
    time_str = setup_train(config, random_seed)
    train(attacker, time_str)