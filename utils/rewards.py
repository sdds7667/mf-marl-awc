import numpy as np

from utils.functions import former_action_view
from utils.run_config import RewardFunction, RunConfig, Reward, ObservationMatrix


class TotalPowerOutputReward(RewardFunction):
    @staticmethod
    def compute(old_reward: np.ndarray, new_reward: np.ndarray, observation_matrix: np.ndarray,
                reward_matrix: np.ndarray,
                config: "RunConfig") -> np.ndarray:
        return np.sum(new_reward) * config.multiplication_factor * np.ones((config.n_turbines,))


class DeltaSumOutputReward(RewardFunction):
    @staticmethod
    def compute(old_reward: Reward, new_reward: Reward, observation_matrix: ObservationMatrix,
                reward_matrix: ObservationMatrix,
                config: "RunConfig") -> Reward:
        delta = (new_reward - old_reward) * config.multiplication_factor
        return np.sum(delta) * np.ones((config.n_turbines,))


class DeltaSumRewardView(RewardFunction):

    @staticmethod
    def compute(old_reward: Reward, new_reward: Reward, observation_matrix: ObservationMatrix,
                reward_matrix: ObservationMatrix,
                config: "RunConfig") -> Reward:
        delta = (new_reward - old_reward) * config.multiplication_factor
        delta_in_reward_view = former_action_view(delta, reward_matrix, config.reward_radius)
        return np.sum(delta_in_reward_view, axis=(1, 2))
