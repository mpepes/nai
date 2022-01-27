# import required packages

import gym 
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


### FUNCTIONS ###

def build_model(height, width, channels, actions):
    '''
    Build a neural network for action preditions.

    @param: height Height of observation space
    @param: width Width of observation space
    @param: channels Depth of observation space
    @param: actions Possible actions to perform in the environment
    '''
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

########################################################

def build_agent(model, actions):
    '''
    Build an agent, i.e. object, which will be exploring environment.
    Parameters might be adjusted to tune performance.

    @param: model Model for predictions
    @param: actions Possible actions to perform in the environment
    '''
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn

if __name__ == "__main__":

    # Define game environment - SpaceInvaders
    env = gym.make('SpaceInvaders-v0')

    # get the shape of observation space
    height, width, channels = env.observation_space.shape

    # get possible actions
    actions = env.action_space.n

    # print possible acctions

    print(env.unwrapped.get_action_meanings())

    # build the model
    model = build_model(height, width, channels, actions)

    # print summarization of the model
    print(model.summary())

    # build an agent
    dqn = build_agent(model, actions)

    # compile, with optimizer ADAM
    dqn.compile(Adam(lr=1e-4))

    # perform training 
    dqn.fit(env, nb_steps=10000, visualize=False)

    # test the model
    scores = dqn.test(env, nb_episodes=2, visualize=True)

    # print rewards for each episode. Higher is better.
    print(np.mean(scores.history['episode_reward']))

    # save weights
    dqn.save_weights('dqn_weights_shorter.h5f')

    
