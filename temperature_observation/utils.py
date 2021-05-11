import numpy as np


def normalize_observation(observation):
    """Normalizes a TemperatureObservation to a flattened array

    Args:
        observation (np.ndarray): Observation provided by TemperatureObservation

    Returns:
        np.ndarray: Flattened observation
    """
    return observation.flatten()


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
