#!/usr/bin/env python3

from __future__ import print_function
from collections import deque

from scratchpad.pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import gym
import argparse

from common import MySQL, Languages, Timer
from helpers import Env, Link

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, deleteDuplicateTransitions, langPair):
        self.gamma = 0.9 #1.0 #0.99
        self.lrn_rate = 3e-4
        self.alpha = 1.0 # 0.7
        self.max_epochs = 200001
        self.eps = 0.1
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 10
        
        self.debug = False
        self.walk = 50
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 100.0 #17.0
        self.cost = -1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = 400 #9999999999

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = [languages.GetLang(langPairList[0]), languages.GetLang(langPairList[1])] 
        #print("self.langs", self.langs)

######################################################################################

######################################################################################
def policy_network(states):
  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("W2", [20, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  p = tf.matmul(h1, W2) + b2
  #print("p", p)
  return p

######################################################################################
def main():
    global state_dim, num_actions
    global TIMER
    TIMER = Timer()

    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.)")
    oparser.add_argument("--language-pair", dest="langPair", required=True,
                         help="The 2 language we're interested in, separated by ,")
    oparser.add_argument("--save-dir", dest="saveDir", default=".",
                         help="Directory that model WIP are saved to. If existing model exists then load it")
    oparser.add_argument("--delete-duplicate-transitions", dest="deleteDuplicateTransitions",
                         default=False, help="If True then only unique transition are used in each batch")
    options = oparser.parse_args()

    sqlconn = MySQL(options.configFile)

    languages = Languages(sqlconn.mycursor)
    params = LearningParams(languages, options.saveDir, options.deleteDuplicateTransitions, options.langPair)

    hostName = "http://vade-retro.fr/"
    #hostName = "http://www.buchmann.ch/"
    #hostName = "http://www.visitbritain.com/"
    env = Env(sqlconn, hostName)

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

    state_dim   = env.observation_space.shape[0]
    num_actions = env.action_space.n

    pg_reinforce = PolicyGradientREINFORCE(sess,
                                        optimizer,
                                        policy_network,
                                        state_dim,
                                        num_actions,
                                        summary_writer=writer,
                                        discount_factor=1.0)

    MAX_EPISODES = 10000
    MAX_STEPS    = 200

    episode_history = deque(maxlen=100)
    for i_episode in range(MAX_EPISODES):

        # initialize
        state = env.reset()
        total_rewards = 0

        for t in range(MAX_STEPS):
            env.render()
            action = pg_reinforce.sampleAction(state[np.newaxis,:])
            next_state, reward, done, _ = env.step(action)

            total_rewards += reward
            reward = -10 if done else 0.1 # normalize reward
            #print("action, next_state, reward, done", action, next_state, reward, done)
            pg_reinforce.storeRollout(state, action, reward)

            state = next_state
            if done: break
        
        pg_reinforce.updateModel()

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(t+1))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        if mean_rewards >= 195.0 and len(episode_history) >= 100:
            print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
            break
        print()

######################################################################################
main()
