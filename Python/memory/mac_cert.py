from rl.memory import RingBuffer, SequentialMemory
import numpy as np

class Macro_Concurrent_Experience_Replay_Memory:
    def __init__(self, buffer_size=1000, window_length=1):
        self._window_length = window_length
        self._buffer_size = buffer_size

        replay_buffer = SequentialMemory(limit=self._buffer_size, window_length=self._window_length)

        # macro_observation (z)
        # self.actions = RingBuffer(self._buffer_size)

        # # macro-action (m)
        # self.rewards = RingBuffer(self._buffer_size)

        # # new macro-observation (z')    (while m is being executed, this is the same as z)
        # self.next_reward = RingBuffer(self._buffer_size)

        # # accumulating reward
        # self.rewards = RingBuffer(self._buffer_size)

        # # Finished
        # self.terminals = RingBuffer(self._buffer_size)

    def sample(self, batch_size):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
        # Returns
            A list of experiences randomly selected
        """
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal):
        raise NotImplementedError()

    def nb_entries(self):
        raise NotImplementedError()
    

    