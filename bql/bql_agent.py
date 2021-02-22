"""
An agent for the IDSGameEnv that implements the BQL algorithm
"""
from typing import Union
import numpy as np
import time
import tqdm
import math
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.training_agents.q_learning.q_agent import QAgent
import os
from bql.utils import bql_f_inv, normal_gamma

class BQLAgent(QAgent):
    """
    A simple implementation of the BQL algorithm.
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(BQLAgent, self).__init__(env, config)
        self.Q_attacker = {}
        self.Q_defender = {}
        self.env.idsgame_config.save_trajectories = False
        self.env.idsgame_config.save_attack_stats = True
        self.initialize_prior()

    def initialize_prior(self):
        for s in range(self.env.num_states):
            for a in range(self.env.num_attack_actions):
                if s not in self.Q_attacker: self.Q_attacker[s] = {}

                # Somewhat unrealistically, we assume here that the attacker has domain knowledge about the initial defense state
                # The initial defense state is accessible from the env like as env.state.defense_values
                # We use this information to give a prior that gives higher expected reward from attacks on places where the defense is low
                target_node_id, _, attack_type, _ = self.env.get_attacker_action((a, None))
                attack_bonus = 10*(np.mean(self.env.state.defense_values[target_node_id]) - self.env.state.defense_values[target_node_id][attack_type])
                self.Q_attacker[s][a] = (self.config.a_mu0 + attack_bonus, self.config.a_lambda0, self.config.a_alpha0, self.config.a_beta0)

        s = 0
        for a in range(self.env.num_defense_actions):
            if s not in self.Q_defender: self.Q_defender[s] = {}

            defense_node_id, _, defense_type, = self.env.get_defender_action((None, a))
            if defense_type < 10:
                defense_bonus = 10 * (np.mean(self.env.state.defense_values[defense_node_id]) -
                                     self.env.state.defense_values[defense_node_id][defense_type])
            else:
                defense_bonus = 0
            self.Q_defender[s][a] = (self.config.d_mu0 + defense_bonus, self.config.d_lambda0, self.config.d_alpha0, self.config.d_beta0)

    def sample_q(self, s):

        # Arrays for holding q samples and corresponding actions
        qs, acts = [], []

        for a, hyp in self.Q_attacker[s].items():
            # Sample from student-t distribution
            st = np.random.standard_t(2 * hyp[2])

            # q sample from t:  m0 + t * (beta / (lamda * alpha))**0.5
            qs.append(hyp[0] + st * (hyp[3] / (hyp[1] * hyp[2])) ** 0.5)
            acts.append(a)

        return np.array(qs), np.array(acts)

    def kl_matched_hyps(self, s, a, r, s_, attacker = True):

        num_samples = self.config.num_mixture_samples

        # Find the action from s_ with the largest mean
        a_ = self.max_mu0_action(s_, attacker=attacker)

        # Parameters for next state-action NG and posterior predictive
        if attacker:
            mu0_, lamda_, alpha_, beta_ = self.Q_attacker[s_][a_]
        else:
            mu0_, lamda_, alpha_, beta_ = self.Q_defender[s_][a_]
        coeff = (beta_ * (lamda_ + 1) / (alpha_ * lamda_)) ** 0.5

        # Sample from student-t, rescale and add mean
        st = np.random.standard_t(2 * alpha_, size=(num_samples,))
        z_samp = mu0_ + st * coeff

        # Dicount and add reward
        z_samp = r + self.config.gamma * z_samp

        # z_sa posterior hyperparameters
        if attacker:
            mu0_sa, lamda_sa, alpha_sa, beta_sa = self.Q_attacker[s][a]
        else:
            mu0_sa, lamda_sa, alpha_sa, beta_sa = self.Q_defender[s][a]

        # z_sa posterior hyperparameters updated for each sample
        mu0_ = (lamda_sa * mu0_sa + z_samp) / (lamda_sa + 1)
        lamda_ = np.array([lamda_sa + 1] * mu0_.shape[0])
        alpha_ = np.array([alpha_sa + 0.5] * mu0_.shape[0])
        beta_ = beta_sa + lamda_sa * (z_samp - mu0_sa) ** 2 / (2 * lamda_sa + 2)

        # Sample mu and tau for each set of updated hyperparameters
        mus, taus = normal_gamma(mu0_, lamda_, alpha_, beta_)

        # MC estimates of moments
        E_tau = np.mean(taus)
        E_mu_tau = np.mean(mus * taus)
        E_mu2_tau = np.mean(mus ** 2 * taus)
        E_log_tau = np.mean(np.log(taus))

        # f^-1(x) where f(x) = log(x) - digamma(x)
        f_inv_term = bql_f_inv(np.log(E_tau) - E_log_tau)

        # Calculate hyperparameters of KL-matched normal gamma
        mu0 = E_mu_tau / E_tau
        lamda = 1 / (1e-12 + E_mu2_tau - E_tau * mu0 ** 2)
        alpha = max(1 + 1e-6, f_inv_term)
        beta = alpha / E_tau

        return mu0, lamda, alpha, beta

    def get_action(self, s, attacker=True, eval=False):
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))

        # Sample q values for each action from current state
        qs, acts = self.sample_q(s)

        max_legal_action_value = float("-inf")
        max_legal_action = float("-inf")
        if attacker:
            for i in range(len(self.Q_attacker[s])):
                if i in legal_actions and qs[i] > max_legal_action_value:
                    max_legal_action_value = qs[i]
                    max_legal_action = i
        else:
            for i in range(len(self.Q_defender[s])):
                if i in legal_actions and self.Q_defender[s][i] > max_legal_action_value:
                                max_legal_action_value = self.Q_defender[s][i]
                                max_legal_action = i
        return max_legal_action

    def max_mu0_action(self, s, attacker=True):

        # Get actions and corresponding hyperparameters of R_sa distribution
        if attacker:
            a_mu0 = [(a, hyp[0]) for (a, hyp) in self.Q_attacker[s].items()]
        else:
            a_mu0 = [(a, hyp[0]) for (a, hyp) in self.Q_defender[s].items()]
        a, mu0 = [np.array(arr) for arr in zip(*a_mu0)]

        return a[np.argmax(mu0)]


    def train(self) -> ExperimentResult:
        """
        Runs the BQL-learning algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        attacker_obs, defender_obs = self.env.reset(update_stats=False)

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                s_idx_a = 0
                defender_state_node_id = 0
                s_idx_d = defender_state_node_id
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    s_idx_a = self.env.get_attacker_node_from_observation(attacker_obs)
                    attacker_action = self.get_action(s_idx_a)

                if self.config.defender:
                    s_idx_d = defender_state_node_id
                    defender_action = self.get_action(s_idx_d, attacker=False)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                reward, obs_prime, done = self.step_and_update(action, s_idx_a, s_idx_d)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Render final frame
            if self.config.render:
                self.env.render(mode="human")

            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps, None, None, lr=self.config.alpha)
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval(episode)

            # Save Q table every <self.config.checkpoint_frequency> episodes
            if episode % self.config.checkpoint_freq == 0:
                self.save_q_table()
                self.env.save_trajectories(checkpoint = True)
                self.env.save_attack_data(checkpoint = True)
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            # Reset environment for the next episode and update game stats
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=True)
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(self.config.num_episodes, log=False)

        # Log and return
        self.log_state_values()

        # Save Q Table
        self.save_q_table()

        # Save other game data
        self.env.save_trajectories(checkpoint = False)
        self.env.save_attack_data(checkpoint = False)
        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        return self.train_result

    def step_and_update(self, action, s_idx_a, defender_state_node_id) -> Union[float, np.ndarray, bool]:
        obs_prime, reward, done, info = self.env.step(action)
        attacker_reward, defender_reward = reward
        attacker_obs_prime, defender_obs_prime = obs_prime
        attacker_action, defender_action = action

        if self.config.attacker:
            s_prime_idx = self.env.get_attacker_node_from_observation(attacker_obs_prime)
            self.bql_update(s_idx_a, attacker_action, attacker_reward, s_prime_idx,attacker=True)
        else:
            s_prime_idx = 0
            self.bql_update(defender_state_node_id, defender_action, defender_reward, s_prime_idx, attacker=False)

        return reward, obs_prime, done


    def bql_update(self, s : int, a : int, r : float, s_prime : int, attacker=True):
        # Update hyperparameters
        hyps = self.kl_matched_hyps(s, a, r, s_prime, attacker=attacker)
        if attacker:
            self.Q_attacker[s][a] = hyps
        else:
            self.Q_defender[s][a] = hyps


    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param log: whether to log the result
        :param train_episode: train episode to keep track of logs and plots
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        attacker_obs, defender_obs = self.env.reset(update_stats=False)

        # Get initial frame
        if self.config.video or self.config.gifs:
            initial_frame = self.env.render(mode="rgb_array")[0]
            self.env.episode_frames.append(initial_frame)

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            attacker_state_values = []
            attacker_states = []
            attacker_frames = []
            defender_state_values = []
            defender_states = []
            defender_frames = []

            if self.config.video or self.config.gifs:
                attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
                attacker_state_values.append(sum(self.Q_attacker[attacker_state_node_id]))
                attacker_states.append(attacker_state_node_id)
                attacker_frames.append(initial_frame)
                defender_state_node_id = 0
                defender_state_values.append(sum(self.Q_defender[defender_state_node_id]))
                defender_states.append(defender_state_node_id)
                defender_frames.append(initial_frame)

            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_state_node_id = 0
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    s_idx_a = self.env.get_attacker_node_from_observation(attacker_obs)
                    attacker_action = self.get_action(s_idx_a, attacker=True, eval=True)

                if self.config.defender:
                    s_idx_d = defender_state_node_id
                    defender_action = self.get_action(s_idx_d, attacker=False, eval=True)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

                # Save state values for analysis later
                if self.config.video and len(self.env.episode_frames) > 1:
                    if self.config.attacker:
                        attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
                        attacker_state_values.append(sum(self.Q_attacker[attacker_state_node_id]))
                        attacker_states.append(attacker_state_node_id)
                        attacker_frames.append(self.env.episode_frames[-1])

                    if self.config.defender:
                        defender_state_node_id = 0
                        defender_state_values.append(sum(self.Q_defender[defender_state_node_id]))
                        defender_states.append(defender_state_node_id)
                        defender_frames.append(self.env.episode_frames[-1])

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            self.eval_attacker_cumulative_reward += episode_attacker_reward
            self.eval_defender_cumulative_reward += episode_defender_reward
            if self.env.state.hacked:
                self.num_eval_hacks += 1
                self.num_eval_hacks_total += 1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_games > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps, update_stats=False, eval = True)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

            if len(attacker_frames) > 1:
                # Save state values analysis for final state
                base_path = self.config.save_dir + "/state_values/" + str(train_episode) + "/"
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                np.save(base_path + "attacker_states.npy", attacker_states)
                np.save(base_path + "attacker_state_values.npy", attacker_state_values)
                np.save(base_path + "attacker_frames.npy", attacker_frames)


            if len(defender_frames) > 1:
                # Save state values analysis for final state
                base_path = self.config.save_dir + "/state_values/" + str(train_episode) + "/"
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                np.save(base_path + "defender_states.npy", np.array(defender_states))
                np.save(base_path + "defender_state_values.npy", np.array(defender_state_values))
                np.save(base_path + "defender_frames.npy", np.array(defender_frames))

            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            # Get initial frame
            if self.config.video or self.config.gifs:
                initial_frame = self.env.render(mode="rgb_array")[0]
                self.env.episode_frames.append(initial_frame)

            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_games > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(self.num_eval_games_total)
            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, update_stats=True, eval=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def log_state_values(self) -> None:
        """
        Utility function for printing the state-values according to the learned Q-function

        :return: None
        """
        if self.config.attacker:
            self.config.logger.info("--- Attacker State Values ---")
            for i in range(len(self.Q_attacker)):
                state_value = sum(self.Q_attacker[i])
                node_id = i
                self.config.logger.info("s:{},V(s):{}".format(node_id, state_value))
            self.config.logger.info("--------------------")

        if self.config.defender:
            self.config.logger.info("--- Defender State Values ---")
            for i in range(len(self.Q_defender)):
                state_value = sum(self.Q_defender[i])
                node_id = i
                self.config.logger.info("s:{},V(s):{}".format(node_id, state_value))
            self.config.logger.info("--------------------")

    def save_q_table(self) -> None:
        """
        Saves Q table to disk in binary npy format

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_q_table.npy"
                self.config.logger.info("Saving Q-table to: {}".format(path))
                np.save(path, self.Q_attacker)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_q_table.npy"
                self.config.logger.info("Saving Q-table to: {}".format(path))
                np.save(path, self.Q_defender)
        else:
            self.config.logger.warning("Save path not defined, not saving Q table to disk")

