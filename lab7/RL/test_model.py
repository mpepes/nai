#authors: Piotr Michałek s19333 & Kibort Jan s19916
import gym 
import random
import numpy as np
from tensorflow.keras.optimizers import Adam

from training_atari import build_agent, build_model


if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')

    # get the shape of observation space
    height, width, channels = env.observation_space.shape

    # get possible actions
    actions = env.action_space.n

    # build the model
    model = build_model(height, width, channels, actions)

    # build an agent
    dqn = build_agent(model, actions)

    # compile, with optimizer ADAM. Optimizer is required for training, but its positional argument
    # and without calling compile weights can not be loaded.
    dqn.compile(Adam(lr=1e-4))

    # load trained weights
    dqn.load_weights('dqn_weights_longer.h5f')

    # Compare scores of random choice with scores of the trained RL
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            action = random.choice([0,1,2,3,4,5])
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

    # RL
    scores = dqn.test(env, nb_episodes=10, visualize=True)
    print(np.mean(scores.history['episode_reward']))


    # It can be observed, that model performing some tactics. It's hiding and heading left side.
    # Performance can be increased by longer training. Model is trained on 1m iterations. Performance peak
    # can be acquired by traning about 40m iterations. But this number is beyond the scope of our hardware.
