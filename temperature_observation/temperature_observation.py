from flatland.envs.observations import ObservationBuilder
from scipy import ndimage
import collections
import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.observations import TreeObsForRailEnv


Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                              'dist_other_target_encountered '
                              'dist_other_agent_encountered '
                              'dist_potential_conflict '
                              'dist_unusable_switch '
                              'dist_to_next_branch '
                              'dist_min_to_target '
                              'num_agents_same_direction '
                              'num_agents_opposite_direction '
                              'num_agents_malfunctioning '
                              'speed_min_fractional '
                              'num_agents_ready_to_depart '
                              'childs '
                              'temperature_other_trains '
                              'temperature_other_stations '
                              'temperature_own_station '
                              'temperature_own_position')


class TemperatureObservation(TreeObsForRailEnv):
    """Observation for the Flatland environment based on thermodynamics

    Args:
        ObservationBuilder (ObservationBuilder): The ObservationBuilder class provided by Flatland.
    """

    def reset(self):
        """Resets the observation, filling the arrays with zeros and

        Returns:
            tuple: The observation and the info dictionary
        """
        super().reset()
        self.temperature_grid = np.zeros(
            (4, self.env.height, self.env.width, len(self.env.agents)))
        return self.temperature_grid, {}

    def relax_temperature(self, handle: int = 0):
        """Propagates the temperature using a Gaussian Filter

        Args:
            handle (int, optional): The train we're providing the observation from. Defaults to 0.

        Returns:
            np.array: The relaxed observation
        """
        temp = self.temperature_grid[:, :, :, handle]
        temp[:3, :, :] = ndimage.gaussian_filter(
            self.temperature_grid[:3, :, :, handle], 1)
        mask = self.env.rail.grid > 0
        temp[:, ~mask] = 0
        return temp

    def get_temperature(self, handle: int = 0) -> (np.ndarray):
        """Builds the actual observation for a single train

        Returns:
            np.ndarray: The observation for a single train
        """
        agent = self.env.agents[handle]
        x_target = agent.target[0]
        y_target = agent.target[1]
        # This sets the temperatures for other 🚂s (which are hot)
        for i, other_agent in enumerate(self.env.agents):
            x = other_agent.position[0] if other_agent.position is not None else None
            y = other_agent.position[1] if other_agent.position is not None else None
            if handle != i:
                self.temperature_grid[0, x, y, i] = 1
                # This heats up others trains' stations
                self.temperature_grid[1, other_agent.target[0],
                                      other_agent.target[1], handle] = 1

        self.temperature_grid[2, x_target, y_target, handle] = 1
        x = agent.position[0] if agent.position is not None else None
        y = agent.position[1] if agent.position is not None else None
        self.temperature_grid[3, x, y, handle] = 1
        return self.relax_temperature(handle)

    def get(self, handle: int = 0) -> Node:
        return self.get_temperature(handle)