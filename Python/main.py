import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from environments.search_env import SearchEnv

def build_model(states, actions):
    model = Sequential()

    model.add(Flatten(input_shape=states))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=130000, window_length=1)
    dqn = DQNAgent(model=model, 
                   memory=memory, 
                   policy=policy, 
                   nb_actions=actions, 
                   nb_steps_warmup=150, 
                   target_model_update=1e-2)

    return dqn

def save_model(model):
    model.save_weights('dqn_robot_weights.h5f', overwrite=True)

def main():
    env = SearchEnv(32, 2)
    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model((1, *states), actions)
    model.summary()

    dqn = build_agent(model, actions)
    dqn.compile(tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
    dqn.fit(env, nb_steps=30000, visualize=True, verbose=1)

    # dqn.test(env, nb_episodes=3, visualize=True)

    result = input("Do you want to save this model? (y/n)")
    if result == "y":
        save_model(dqn)

if __name__ == "__main__":
    main()