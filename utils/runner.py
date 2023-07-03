import functools
import os
import pathlib
import time
from typing import List, Callable, Tuple

import tensorflow as tf
import numpy as np

from alg.q_learning import MFQ
from utils.functions import former_action_view, observation_space_from_state
from utils.run_config import RunConfig
from wind_farm_gym import WindFarmEnv

action_count = 3


def action_space_from_env(env: WindFarmEnv) -> np.ndarray:
    """
    Get the action space from the environment.
    :param env: the environment
    :return: the action space
    """
    return np.zeros((env.n_turbines, action_count))


def former_action_space(radius: Tuple[int, int]) -> np.ndarray:
    """
    Use a specialized former action probability array that is neighbour aware.
    It is based on the same mathematics as the observation matrix.

    :param radius: the radius of the view
    :return: the former action probability space
    """
    return np.zeros((radius[0] * 2 + 1, radius[1] * 2 + 1))


def decode_action(action: np.ndarray) -> np.ndarray:
    """
    Decode the action into yaw angles.

    The encoding is
        0: -1 / pitch left as much as possible
        1: 0 / no change
        2: 1 / pitch right as much as possible
    :param action: the action to decode.
    :return: the yaw angles
    """
    return action - 1.0


def feature_space_from_state(state: List[float], build_functions: List[Callable[[np.array], np.array]]) -> np.ndarray:
    """
    Build the features for each turbine in the environment.
    Feature space: Wind direction at the turbine, wind speed at the turbine, yaw angle of the turbine
    """
    return np.stack([fn(state) for fn in build_functions], axis=0)


class Runner:

    def __init__(self, config: RunConfig):
        self.wind_farm_env = WindFarmEnv(
            turbine_layout=config.layout,
            desired_yaw_boundaries=(-40, 40),
        )
        self.config = config
        tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=tf_config)
        tf.compat.v1.disable_eager_execution()
        self.mf_q = MFQ(self.sess, "Test", 0,
                        self.wind_farm_env,
                        sub_len=config.step_count,
                        radius=config.radius,
                        update_every=config.update_every,
                        batch_size=config.batch_size,
                        memory_size=config.buffer_size,
                        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.start = 0
        self.BASE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.dir = self.BASE_DIR / f'data'
        self.model_dir = self.dir / "models" / f"{config.run_name}/run-{config.run_id}/"

        self.log_directory = self.dir / "logs" / f"{config.run_name}/run-{config.run_id}/"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory, exist_ok=True)

        self.tensorboard_summary = tf.compat.v1.summary.FileWriter(self.log_directory.as_posix())
        self.final_rewards_tensor = tf.compat.v1.placeholder(tf.float32, shape=(None,), name="FinalRewards")
        self.final_rewards_summary = tf.compat.v1.summary.scalar("final_rewards",
                                                                 tf.reduce_sum(self.final_rewards_tensor))

        self.mean_rewards_tensor = tf.compat.v1.placeholder(tf.float32, shape=(None,), name="MeanRewards")
        self.mean_rewards_summary = tf.compat.v1.summary.scalar("mean_rewards",
                                                                tf.reduce_sum(self.mean_rewards_tensor))

        self.loss = tf.compat.v1.placeholder(tf.float32, shape=(None,), name="Loss")
        self.loss_summary = tf.compat.v1.summary.scalar("loss", tf.reduce_sum(self.loss))
        self.summary_op = tf.compat.v1.summary.merge_all()

        if os.path.exists(self.model_dir):
            # get the latest model file
            path = pathlib.Path(self.model_dir)
            files = list(path.glob('*.index'))
            # Extract only the number part of the filename
            files = [int(f.name.split("_")[-1].split('.')[0]) for f in files]
            files.sort()
            if len(files) != 0:
                self.mf_q.load(dir_path=self.model_dir, step=files[-1])
                self.start = files[-1]

        def linear_decay(epoch, x, y):
            min_v, max_v = y[0], y[-1]
            start, end = x[0], x[-1]

            if epoch == start:
                return min_v

            eps = min_v

            for i, x_i in enumerate(x):
                if epoch <= x_i:
                    interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
                    eps = interval * (epoch - x[i - 1]) + y[i - 1]
                    break

            return eps

        self.linear_decay = functools.partial(linear_decay,
                                              x=[0, int(config.episode_count * 0.4), config.episode_count, 100_000],
                                              y=[1, 0.4, 0.1, 0.1]
                                              )

        self.feature_functions = self.build_feature_functions()
        self.observation_matrix = self.build_observation_matrix(config.radius)
        self.reward_matrix = self.build_observation_matrix(config.reward_radius)

    def optimize(self):
        for k in range(self.start, self.config.episode_count):
            eps = self.linear_decay(k)
            try:
                if k % self.config.save_every == 0:
                    self.mf_q.save(self.model_dir, k)

                # increase the max number of steps by 10 every 100 iterations
                round_time = time.time()
                mean, total, final, loss = self.episode(eps, k)
                if self.config.train:
                    results = self.sess.run(self.summary_op,
                                            feed_dict={
                                                self.final_rewards_tensor: final,
                                                self.loss: np.array([loss]),
                                                self.mean_rewards_tensor: mean
                                            })
                    self.tensorboard_summary.add_summary(results, k)
                    self.tensorboard_summary.flush()
                if k % self.config.update_every == 0:
                    print(
                        f"Final Reward : {final}, Mean Reward : {mean}, Total Reward : {total}, Loss : {loss}, Time : {time.time() - round_time}")
                print(f"Final Reward: {np.sum(final)}. Time: {time.time() - round_time}")

            except Exception as e:
                print("Unknown graph execution error: {}".format(e))

    def build_feature_functions(self) -> List[Callable[[List[float]], np.array]]:
        """
        Extract the features related to each function from the environment.
        Since the state is arranged as [yaw for each turbine, wind speed for each turbine, wind direction for each turbine],
        we can just extract the features from the state, and stack them together.

        :param env: the environment
        :return: the feature functions
        """
        env = self.wind_farm_env

        def extract_from_state(turbine_id, state) -> np.array:
            return np.array(
                [state[turbine_id], state[env.n_turbines + turbine_id * 2], state[env.n_turbines + turbine_id * 2 + 1]])

        return [functools.partial(extract_from_state, i) for i in range(env.n_turbines)]

    def build_observation_matrix(self, radius: Tuple[int, int], include_self=True, ) -> np.array:
        """
        Build the observation functions from the environment.
        Each turbine has a grid view around itself, with shape (radius_x * 2 + 1, radius_y * 2 + 1, 3),
        Channels:
            1. wind speed
            2. wind direction
            3. turbine yaw angle
        If there are multiple turbines in a single square, take the average of the values.


        :param radius: radius of the grid view
        :param include_self: whether to include the turbine itself in the observation.
        :return: the observation functions, one that builds the view for each turbine
        """
        env = self.wind_farm_env
        cell_width = self.config.cell_width

        feature_array = np.zeros((env.n_turbines * (radius[1] * 2 + 1) * (radius[0] * 2 + 1), env.n_turbines))

        env_layout = env.turbine_layout
        for i in range(0, env.n_turbines):
            for t in range(0, env.n_turbines):
                if i == t and not include_self:
                    continue
                dx = ((env_layout[0][t] - env_layout[0][i]) / cell_width) + radius[1]
                dy = ((env_layout[1][t] - env_layout[1][i]) / cell_width) + radius[0]

                if dy < 0 or dy >= (radius[0] * 2 + 1) or dx < 0 or dx >= (radius[1] * 2 + 1):
                    continue
                feature_array[
                    (radius[0] * 2 + 1) * (radius[1] * 2 + 1) * i + int(dy) * (radius[1] * 2 + 1) + int(dx), t] = 1
        return feature_array.T

    def episode(self, eps, episode_id) -> Tuple[float, float, float, float]:
        state = self.wind_farm_env.reset()

        total_reward = 0
        mean_reward = 0
        act = np.ones((self.wind_farm_env.n_turbines,))
        former_act_prob = former_action_view(act, self.observation_matrix, self.config.radius)
        state, reward, _, _ = self.wind_farm_env.step(decode_action(act))
        if np.isnan(reward).any():
            raise ValueError("Reward contains NaN")
        old_reward = reward
        old_rewards = np.zeros((5, self.wind_farm_env.n_turbines))
        max_reward = np.sum(reward)

        for i in range(self.config.step_count):
            feature = feature_space_from_state(state, self.feature_functions)
            observation = observation_space_from_state(feature, self.observation_matrix, self.config.radius)

            # Epsilon greedy exploration
            if self.config.train and eps > np.random.random():
                action = np.random.randint(action_count, size=(self.wind_farm_env.n_turbines,))
            else:
                action = self.mf_q.act(state=[observation, feature], prob=former_act_prob)

            next_observation, reward, _, _ = self.wind_farm_env.step(decode_action(action))
            next_observation = np.array(next_observation)
            if np.isnan(reward).any():
                raise ValueError("Reward contains NaN")

            if np.isnan(next_observation).any():
                raise ValueError("Observation contains NaN")

            rwd = self.config.reward.compute(old_reward, reward, self.observation_matrix, self.reward_matrix,
                                             self.config)
            old_reward = reward

            old_rewards[i % 5] = rwd

            buffer = {
                'state': [observation, feature],
                'acts': action,
                'rewards': rwd,
                'prob': former_act_prob,
                'ids': np.arange(self.wind_farm_env.n_turbines),
            }

            former_act_prob = former_action_view(action, self.observation_matrix, self.config.radius)
            if self.config.train:
                self.mf_q.flush_buffer(**buffer)

            mean_reward = mean_reward + (reward - mean_reward) / (i + 1)
            total_reward += reward
            state = next_observation
            if self.config.render:
                self.wind_farm_env.render()
        if episode_id % self.config.update_every == 0:
            print(
                f'Round {episode_id} complete., Reward: {max_reward}, Mean Reward = {np.sum(mean_reward)}), Eps: {eps}')
            print("Final Reward: ", np.sum(reward))

        if self.config.train:
            loss, q = self.mf_q.train()
        else:
            loss = 0
        return mean_reward, total_reward, reward, loss
