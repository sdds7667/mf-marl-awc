from dataclasses import dataclass

from typing import List, Tuple, Type
from abc import abstractmethod

import numpy as np

"""Shape=[Number of turbines]"""
Features = np.ndarray

"""Shape=[(Radius.x * 2 + 1) * (Radius.y * 2 + 1), Number of turbines]
When multiplied with a feature vector, it will mask out the information that is not needed for each cell, and sum up the rest. 
"""
ObservationMatrixPerTurbine = np.ndarray

"""Shape=[ObservationMatrixPerTurbine, Number of turbines]. """
ObservationMatrix = np.ndarray

"""Shape=[Number of turbines]"""
Reward = np.ndarray


class RewardFunction:

    @staticmethod
    @abstractmethod
    def compute(old_reward: Reward, new_reward: Reward, observation_matrix: ObservationMatrix,
                reward_matrix: ObservationMatrix,
                config: "RunConfig") -> Reward:
        pass


@dataclass
class RunConfig:
    layout: Tuple[List[int], List[int]]
    reward: Type[RewardFunction]
    run_name: str
    run_id: int

    episode_count: int = 4_000
    step_count: int = 150
    cell_width: int = 750
    radius: Tuple[int, int] = (1, 1)
    reward_radius: Tuple[int, int] = (1, 1)
    render: bool = False
    buffer_size: int = 1_000_000
    tau: float = 0.05
    learning_rate: float = 1e-5
    update_every: int = 5
    batch_size: int = 64
    neighbor_aware_former_actions: bool = True  # A version that does not provide the former actions is not implemented

    save_every: int = 30
    multiplication_factor: int = 100_000
    train: bool = True

    @property
    def n_turbines(self):
        return len(self.layout[0])
