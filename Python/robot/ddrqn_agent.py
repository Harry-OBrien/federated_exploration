from macro_observation.teammate_detection import Teammate_Detector
from macro_observation.mapping import Local_Map
from macro_observation.localisation import Localisation
from macro_observation.goal_extraction import Goal_Extractor

from navigation_controller.low_level_controller import Navigation_Controller

from ..memory.mac_cert import Macro_Concurrent_Experience_Replay_Memory

from ..marlgrid.agents import GridAgentInterface

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Flatten, Conv2D, Input, LSTM, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from rl.core import Agent
from rl.policy import BoltzmannQPolicy

import numpy as np

class DDRQN_Agent(Agent):
    def __init__(self,
        start_pos,
        start_orientation, 
        observation_space, 
        action_space, 
        learning_config={}, 
        env_interface=None,
        num_agents=3
    ):
        # Environment interface
        self._env_interface = env_interface

        # Macro observations
        self._localisation = Localisation(start_pos, start_orientation)
        self._teammate_detector = Teammate_Detector(num_agents)
        self._mapping = Local_Map()
        self._goal_extraction = Goal_Extractor(frontier_width=5)

        self._current_goal_location = None
        self._last_goal_location = None

        self._last_information_exchange = [None] * num_agents

        # Macro Actions
        self._navigation_ctrl = Navigation_Controller(self._localisation.get_state(), self._map.get_maps())

        # Mac-CERT
        self._memory = Macro_Concurrent_Experience_Replay_Memory(buffer_size=10_000, window_length=10)

        # model
        self._model = self._create_model(learning_config["learning_rate"])
        self._target_model = self._create_model(learning_config["learning_rate"])
        self._target_model.set_weights(self._model.get_weights())

        # Reset the agent
        self.reset_states(start_pos)

    def _create_model(self, lr):
        model = self.layers
        model.compile(loss="mse", optimize=Adam(learning_rate=lr), metrics="accuracy")

        return model

    ####################### MARK: KERAS_RL_CORE IMPLEMENTATION #######################
    def reset_states(self, start_pos):
        """Resets all internally kept states after an episode is completed.
        """
        self._map.reset_map()
        self._current_goal_location = None
        self._last_goal_location = None

    # action step
    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        # our input sensors is the observation. We pass this to the mac_obs module and 
        # the navigation controller

        nearby_agents = self._teammate_detector.find_from(observation)
        for i, agent_exists in enumerate(nearby_agents):
            if not agent_exists:
                continue

            shared_info = self._env_interface.get_shared_info_from_agent(i)

            # There is a chance that we have a comms failure when interracting with another agent. Check for that here
            if shared_info is not None:
                self._last_information_exchange[i] = shared_info

        position, orientation = self._localisation.get_position(observation)
        maps, area_explored = self._mapping.get_local_maps()

        self._navigation_ctrl.update_maps(maps)
        self._navigation_ctrl.update_location(position, orientation)

        # If we need to generate a new goal
        if self._current_goal_location is None or position == self._current_goal_location:
            self._last_goal_location = self._current_goal_location

            potential_goals = self._goal_extraction.generate_goals(maps)
            goal_choice = self._model.predict(position, nearby_agents, self._last_information_exchange, area_explored, maps, potential_goals)

            self._navigation_ctrl.set_goal(goal_choice)

        return self._navigation_ctrl.move()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    # Train
    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.

        # Returns
            A list of the model's layers
        """
        map_input = Input(shape=(32, 32, 3, ), name="map_input") # 3, 32x32 binary feature maps
        macro_obs_input = Input(shape=(3, ), name="macro_observations_input") # 3 values, robot_x, robot_y and %complete

        # First branch is convolutional model to analyse map input
        x = Conv2D(filters=8, kernel_size=(4,4), strides=(2, 2), name="C1")(map_input)
        x = LeakyReLU()(x)
        x = Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2),  name="C2")(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=16, kernel_size=(2,2), strides=(2, 2),  name="C3")(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(32,  name="F1")(x)
        x = LeakyReLU()(x)
        x = Dense(10,  name="F2")(x)
        x = LeakyReLU()(x)
        x = Model(inputs=map_input, outputs=x)

        # Second branch is a FCL to analyse macro observations
        y = Dense(64, name="F3")(macro_obs_input)
        y = LeakyReLU()(y)
        y = Model(inputs=macro_obs_input, outputs=y)

        combined = concatenate([x.output, y.output])

        z = Dense(64,  name="F4")(combined)
        z = LeakyReLU()(z)
        z = Dense(64,  name="F5")(z)
        z = LeakyReLU()(z)
        z = Reshape((1, 64), input_shape=(64,))(z)
        z = LSTM(64)(z)
        z = LeakyReLU()(z)
        z = Dense(64,  name="F6")(z)
        z = LeakyReLU()(z)
        model_output = Dense(5, activation='linear')(z)
        
        inputs = [map_input, macro_obs_input]
        return tf.keras.Model(inputs, model_output, name="DEP")