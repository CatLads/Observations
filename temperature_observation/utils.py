from flatland.envs.observations import TreeObsForRailEnv
import numpy as np


def format_action_prob(action_probs):
    """Generates a probability string 

    Args:
        action_probs (np.ndarray): action probabilities array

    Returns:
        string: string of probabilities to print out
    """
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def normalize_observation(observation):
    """
    This function normalizes the observation used by the RL algorithm
    """
    return np.transpose(observation, (1, 2, 0)).flatten()

