import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print "Reward for this episode was: %d"%(reward_sum)
        reward_sum = 0
        env.reset()


        