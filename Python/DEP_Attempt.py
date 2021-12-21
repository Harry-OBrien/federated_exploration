import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Flatten, Conv2D, Input, LSTM, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np

def DEP():
    map_input = Input(shape=(32, 32, 3), name="map_input") # 3, 32x32 binary feature maps
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

def main():
    model = DEP()
    model.summary()
    tf.keras.utils.plot_model(model, "DEP.png", show_shapes=True)


if __name__ == "__main__":
    main()