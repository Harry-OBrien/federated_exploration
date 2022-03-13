"""
The Mapping module performs both mapping and map merging. 
Each robot updates its local map, 𝓂𝑖 ∈ M, and the area explored, |𝐸𝑖 |, using its corresponding sensors.
A map is defined by a four-channel image, where each channel is a binary feature map of the explored space, obstacles, 
observed robot positions, and goal candidates. 
Local maps are merged when robots are within 𝑑𝑠 to provide updates on explored regions.
A global map, M, is generated for centralized training, by combining all robots’ local maps at each timestep.
"""
class Local_Map:
    def __init__(self, map_starting_size):
        self._map_size = map_starting_size
        self.reset_maps()

    def reset_maps(self):
        """
        Resets all maps to blank values
        """
        self._maps = {"explored_space":  [[0] * self._map_size for _ in range(self._map_size)],
                      "obstacles":       [[0] * self._map_size for _ in range(self._map_size)],
                      "robot_positions": [[0] * self._map_size for _ in range(self._map_size)],
                      "goal_candidates": [[0] * self._map_size for _ in range(self._map_size)]}

    def get_maps(self):
        """
        Gets the agent's local maps

        # Returns
        4 binary feature maps of the explored space, obstacles, observed robot positions, and goal candidates
        """
        return self._maps

    def update_goal_candidates(self, goal_candidates):
        self._maps["goal_candidates"] = [[0] * self._map_size for _ in range(self._map_size)]
        for candidate in goal_candidates:
            self._maps["goal_candidates"][candidate] = 1

    def update_local_map(self, primitive_obs, obs_start, obs_end):
        """
        Updates the agent's local maps with the primitive observations from the agent's sensors

        # Argument
        primitive_obs - The observations from the agent's sensors as 3 binary feature maps of explored space, obstacles and observed robot 
        positions.
        """
        x_start, y_start = obs_start
        x_end, y_end = obs_end

        for key in self._maps.keys():
            if key == "goal_candidates":
                continue

            self._maps[key][y_start:y_end, x_start:x_end] |= primitive_obs[key]

    def union_map(self, other_map):
        """
        Merges the agent's local map and the map of another agent.
        This is really easy as the agent's have perfect observation and localisation skills, so it's a case of putting a 1
        in the maps where there wasn't one

        # Argument
        map_2 The other agent's maps as 4 binary feature maps
        """
        for key in self._maps.keys():
            self._maps[key] |= other_map[key]