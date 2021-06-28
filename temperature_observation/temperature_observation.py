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
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """
        agent_temperature = self.get_temperature(handle)
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle,
                  " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(
            *position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()
        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                     dist_other_agent_encountered=0, dist_potential_conflict=0,
                                     dist_unusable_switch=0, dist_to_next_branch=0,
                                     dist_min_to_target=distance_map[
                                         (handle, *position,
                                          agent.direction)],
                                     num_agents_same_direction=0, num_agents_opposite_direction=0,
                                     num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                     speed_min_fractional=agent.speed_data['speed'],
                                     num_agents_ready_to_depart=0,
                                     childs={},
                                     temperature_other_trains=agent_temperature[0][position],
                                     temperature_other_stations=agent_temperature[1][position],
                                     temperature_own_station=agent_temperature[2][position],
                                     temperature_own_position=agent_temperature[3][position])

        #print("root node type:", type(root_node_observation))

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(
                    position, branch_direction)

                branch_observation, branch_visited = \
                    self._explore_branch(
                        handle, new_cell, agent_temperature, branch_direction, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]
                                             ] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def _explore_branch(self, handle, position, agent_temperature, direction, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []
        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_terminal = False
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(
                    position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    # Attention this counts to many agents as a few might be going off on a switch.
                    other_agent_opposite_direction += self.location_has_agent[position]

                # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(
                *position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(
                    self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth -
                                    1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(
                            self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(
                            self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(
                            self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get(
            )[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get(
            )[handle, position[0], position[1], direction]

        # TreeObsForRailEnv.Node
        node = Node(dist_own_target_encountered=own_target_encountered,
                    dist_other_target_encountered=other_target_encountered,
                    dist_other_agent_encountered=other_agent_encountered,
                    dist_potential_conflict=potential_conflict,
                    dist_unusable_switch=unusable_switch,
                    dist_to_next_branch=dist_to_next_branch,
                    dist_min_to_target=dist_min_to_target,
                    num_agents_same_direction=other_agent_same_direction,
                    num_agents_opposite_direction=other_agent_opposite_direction,
                    num_agents_malfunctioning=malfunctioning_agent,
                    speed_min_fractional=min_fractional_speed,
                    num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                    childs={},
                    temperature_other_trains=agent_temperature[0][position],
                    temperature_other_stations=agent_temperature[1][position],
                    temperature_own_station=agent_temperature[2][position],
                    temperature_own_position=agent_temperature[3][position])
        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(
            *position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(
                    position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          agent_temperature,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]
                            ] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          agent_temperature,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]
                            ] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited

