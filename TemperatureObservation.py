from flatland.envs.observations import ObservationBuilder
from scipy import ndimage
import numpy as np


class TemperatureObservation(ObservationBuilder):
    def reset(self):
        self.rail_obs = np.zeros(
            (self.env.height, self.env.width, len(self.env.agents)))
        return self.rail_obs, {}

    def relax_temperature(self, handle: int = 0):
        """
        Relax the temperature
        """
        relaxed = ndimage.gaussian_filter(self.rail_obs[:, :, handle], 1)
        mask = self.env.rail.grid > 0
        relaxed[~mask] = np.inf
        return relaxed

    def get(self, handle: int = 0) -> (np.ndarray):
        agent = self.env.agents[handle]
        x_target = agent.target[0]
        y_target = agent.target[1]
        # This sets the temperatures for other 🚂s (which are hot)
        for i, other_agent in enumerate(self.env.agents):
            x = other_agent.position[0] if other_agent.position is not None else None
            y = other_agent.position[1] if other_agent.position is not None else None
            if handle != i:
                self.rail_obs[x, y, i] = 1
                # This heats up others trains' stations
                self.rail_obs[other_agent.target[0],
                              other_agent.target[1], handle] = 0.5

        self.rail_obs[x_target, y_target, handle] = -1

        self.relax_temperature(handle)
        return self.rail_obs[:, :, handle]
