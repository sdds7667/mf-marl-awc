from typing import Tuple

import numpy as np


def observation_space_from_state(feature_state: np.array,
                                 features_array: np.array,
                                 radius: Tuple[int, int]) -> np.ndarray:
    """
    Build the view for each turbine in the environment.

    :param feature_state: the features for each turbine
    :param features_array: the features for each square, for each turbine
    :param radius: tuple that contains the radius of the view
    :return: view for each turbine
    """
    return np.dot(feature_state.T, features_array).T.reshape(feature_state.shape[0], radius[0] * 2 + 1,
                                                             radius[1] * 2 + 1, 3)


def former_action_view(actions: np.array, neighbour_matrix: np.array, radius: Tuple[int, int]) -> np.array:
    """
    Build the former action probability view for each turbine.
    :param actions: the actions for each turbine
    :param neighbour_matrix: the neighbouring multiplication matrix
    :param radius: the radius of the view
    """
    return np.dot(actions.T, neighbour_matrix).T.reshape(neighbour_matrix.shape[0], radius[0] * 2 + 1,
                                                         radius[1] * 2 + 1)
