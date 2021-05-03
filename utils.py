def normalize_observation(observation):
    """Normalizes a TemperatureObservation to a flattened array

    Args:
        observation (np.ndarray): Observation provided by TemperatureObservation

    Returns:
        np.ndarray: Flattened observation
    """
    return observation.flatten()
