#!/usr/bin/env python3

from __future__ import print_function
from collections import deque

from scratchpad.pg_reinforce import PolicyGradientREINFORCE
import os
import sys
import tensorflow as tf
import numpy as np
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
class Transition:
    def __init__(self, currURLId, nextURLId, langRequested, langIds, langFeatures):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.langRequested = langRequested 
        self.langIds = langIds 
        self.langFeatures = langFeatures 

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

######################################################################################
class Candidates:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.dict = {} # parent lang -> links[]

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params, self.env)

        for key, value in self.dict.items():
            #print("key", key, value)
            ret.dict[key] = value.copy()

        return ret
    
    def AddLink(self, link):
        langId = link.parentNode.lang
        if langId not in self.dict:
            self.dict[langId] = []
        self.dict[langId].append(link)
        
    def AddLinks(self, node, visited, params):
        #print("   currNode", curr, currNode.Debug())
        newLinks = node.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

    def Pop(self, action):
        links = self.dict[action]
        assert(len(links) > 0)

        idx = np.random.randint(0, len(links))
        link = links.pop(idx)

        # remove all links going to same node
        for otherLinks in self.dict.values():
            otherLinksCopy = otherLinks.copy()
            for otherLink in otherLinksCopy:
                if otherLink.childNode == link.childNode:
                    otherLinks.remove(otherLink)

        return link

    def HasLinks(self, action):
        if action in self.dict and len(self.dict[action]) > 0:
            return True
        else:
            return False

    def CountLinks(self):
        ret = 0
        for links in self.dict.values():
            ret += len(links)
        return ret

    def GetCounts(self):
        ret = np.zeros([self.env.maxLangId + 1])
        for key, value in self.dict.items():
            ret[key] = len(value)
        return ret


    def RandomLink(self):
        while True:
            langs = list(self.dict)
            action = np.random.choice(langs)
            #print("action", action)
            if self.HasLinks(action):
                return self.Pop(action)
        raise Exception("shouldn't be here")

    def Debug(self):
        ret = ""
        for lang in self.dict:
            ret += "lang=" + str(lang) + ":"
            links = self.dict[lang]
            for link in links:
                ret += " " + link.parentNode.url + "->" + link.childNode.url
        return ret
    
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
def GetNextState(env, params, currNode, action, visited, candidates):
    #print("action", action)
    #print("candidates", candidates.Debug())
    #if action == 0:
    #    stopNode = env.nodes[0]
    #    link = Link("", 0, stopNode, stopNode)
    #elif not candidates.HasLinks(action):
    randomNode = False
    if action == 0 or not candidates.HasLinks(action):
        numLinks = candidates.CountLinks()
        #print("numLinks", numLinks)
        #stopNode = env.nodes[0]
        #link = Link("", 0, stopNode, stopNode)

        if numLinks > 0:
            #print("action", action, candidates.Debug())
            link = candidates.RandomLink()
            #print("link1", link.childNode.Debug())
            randomNode = True
        else:
            stopNode = env.nodes[0]
            link = Link("", 0, stopNode, stopNode)
            #print("link2", link)
    else:
        link = candidates.Pop(action)

    assert(link is not None)
    nextNode = link.childNode
    #print("   nextNode", nextNode.Debug())

    if nextNode.urlId == 0:
        #print("   stop")
        reward = 0.0
    elif nextNode.alignedNode is not None and nextNode.alignedNode.urlId in visited:
        reward = params.reward
        #print("   visited", visited)
        #print("   reward", reward)
        #print()
    else:
        #print("   non-rewarding")
        reward = params.cost

    return link, reward

######################################################################################
def Trajectory(params, env, pg_reinforce):
    MAX_STEPS    = 200
    # initialize
    visited = set()
    langsVisited = np.zeros([env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)

    node = env.nodes[sys.maxsize]
    langsVisited[node.lang] += 1
    total_rewards = 0

    for numSteps in range(MAX_STEPS):
        candidateCounts = candidates.GetCounts()
        #print("candidateCounts", candidateCounts, langsVisited)
        state = np.concatenate((langsVisited, candidateCounts), axis=0)
        #print("state", state)

        action = pg_reinforce.sampleAction(state[np.newaxis,:])
        link, reward = GetNextState(env, params, node, action, visited, candidates)
        print("action", action, link.childNode.Debug(), reward)

        done = False
        if link.childNode.urlId == 0:
            done = True
        if len(visited) > params.maxDocs:
            done = True

        total_rewards += reward
        reward = -10 if done else 0.1 # normalize reward
        #print("action, next_state, reward, done", action, next_state, reward, done)
        pg_reinforce.storeRollout(state, action, reward)

        node = link.childNode
        if done: break

    return numSteps, total_rewards

def Train(params, env, pg_reinforce):
    MAX_EPISODES = 10000

    episode_history = deque(maxlen=100)
    for i_episode in range(MAX_EPISODES):
        numSteps, total_rewards = Trajectory(params, env, pg_reinforce)

        pg_reinforce.updateModel()

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(numSteps+1))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        if mean_rewards >= 195.0 and len(episode_history) >= 100:
            print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
            break
        print()

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
    state_dim = (env.maxLangId + 1) * 2
    num_actions = params.NUM_ACTIONS

    #env_name = 'CartPole-v0'
    #env = gym.make(env_name)
    #state_dim   = env.observation_space.shape[0]
    #num_actions = env.action_space.n

    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format("hh"))

    pg_reinforce = PolicyGradientREINFORCE(sess,
                                        optimizer,
                                        policy_network,
                                        state_dim,
                                        num_actions,
                                        summary_writer=writer,
                                        discount_factor=1.0)

    Train(params, env, pg_reinforce)

######################################################################################
main()
