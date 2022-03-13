import numpy as np
from gym import spaces
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.util import wrappers, agent_selector

def env(**kwargs):
    env = raw_env(**kwargs)
    env = OrderEnforcingWrapper(env)
    return env

def raw_env(AECEnv):
    def __init__(self, n_agents=1, percent_fill=50, random_start_pos=True, seed=None, map_size=32):
        self.n_agents=n_agents
        self.percent_fill=percent_fill
        self.random_start_pos=random_start_pos
        self.seed(seed)
        self.map_size=map_size

        assert self.n_agents > 0
        assert self._map_size > 0

        # (up, right, down, left, no_move) -> 5
        self.action_space = spaces.Discrete(5)
        # 4 binary feature maps:
        #   explored space
        #   obstacles
        #   observed robot positions
        #   (goal candidates) not implemented yet
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,None,None))

        self.reset()

    def step(self, action):
        '''
        Receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary, and info dictionary,
        where each dictionary is keyed by the agent.
        '''
        for key, agent_action in action.items():
            pass



    def reset(self):
        '''
        Resets the environment to a starting state.
        '''
        raise NotImplementedError

    def seed(self, seed=None):
        '''
        Reseeds the environment (making the resulting environment deterministic).
        `reset()` must be called after `seed()`, and before `step()`.
        '''
        self.np_random, _ = seeding.np_random(seed)

    def observe(self, agent):
        '''
        Returns the observation an agent currently can make. `last()` calls this function.
        '''
        raise NotImplementedError

    def render(self, mode='human'):
        '''
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        '''
        raise NotImplementedError

    def state(self):
        '''
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        '''
        raise NotImplementedError('state() method has not been implemented in the environment {}.'.format(self.metadata.get('name', self.__class__.__name__)))

    def close(self):
        '''
        Closes the rendering window, subprocesses, network connections, or any other resources
        that should be released.
        '''
        pass

    def observation_space(self, agent):
        '''
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        '''

        return self.observation_space

    def action_space(self, agent):
        '''
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        '''

        return self.action_spaces[agent]