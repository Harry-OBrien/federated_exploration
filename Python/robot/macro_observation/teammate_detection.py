# The teammate detection module outputs a Boolean value, delta-sub-i, to indicate if a teammate is 
# observed within sensing range d-sub-s
class Teammate_Detector:

    def __init__(self, world_delegate, num_agents):
        self._delegate = world_delegate
        self._num_agents = num_agents

    def attempt_local_detection(self, map_slice):
        """
        Takes the sensing range of the agent and returns all visible agents in this range (including ourself)

        # Argument
        the idices for where we want to slice the map/our sensing range

        # Returns
        Boolean list of agents in range where idx == agent_id
        """
        # get IDs of agents in map
        agents = self._delegate.get_agents_in(map_slice)

        # Create a boolean list of whether agents are in range
        bool_map = [False] * self._num_agents

        # Occupy list
        for agent in agents:
            agent_id = agent.get_id()
            bool_map[agent_id] = True

        return bool_map

