#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import pylab as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from common import MySQL, Languages, Timer
from helpers import Env, Link

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, deleteDuplicateTransitions, langPair):
        self.gamma = 0.99
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 20001
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
        self.maxDocs = 9999999999

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = [languages.GetLang(langPairList[0]), languages.GetLang(langPairList[1])] 
        #print("self.langs", self.langs)

######################################################################################
def NumParallelDocs(env, visited):
    ret = 0
    for urlId in visited:
        node = env.nodes[urlId]
        #print("node", node.Debug())

        if node.alignedNode is not None and node.alignedNode.urlId in visited:
            ret += 1

    return ret

######################################################################################
def naive(env, maxDocs, params):
    ret = []
    todo = []
    todo.append(env.rootNode)

    visited = set()
    langsVisited = {}

    while len(todo) > 0 and len(visited) < maxDocs:
        node = todo.pop(0)
        #print("node", node.Debug())
        
        if node.urlId not in visited:
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def balanced(env, maxDocs, params):
    ret = []
    visited = set()
    langsVisited = {}
    langsTodo = {}

    startNode = env.nodes[sys.maxsize]
    #print("startNode", startNode.Debug())
    assert(len(startNode.links) == 1)
    link = next(iter(startNode.links))

    while link is not None and len(visited) < maxDocs:
        node = link.childNode
        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)
    
            for link in node.links:
                #print("   ", childNode.Debug())
                AddTodo(langsTodo, visited, link)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

        link = PopLink(langsTodo, langsVisited, params)

    return ret

def PopLink(langsTodo, langsVisited, params):
    sum = 0
    # any nodes left to do
    for links in langsTodo.values():
        sum += len(links)
    if sum == 0:
        return None
    del sum

    # sum of all nodes visited
    sumAll = 0
    sumRequired = 0
    for lang, count in langsVisited.items():
        sumAll += count
        if lang in params.langIds:
            sumRequired += count
    sumRequired += 0.001 #1
    #print("langsVisited", sumAll, sumRequired, langsVisited)

    probs = {}
    for lang in params.langIds:
        if lang in langsVisited:
            count = langsVisited[lang]
        else:
            count = 0
        #print("langsTodo", lang, nodes)
        prob = 1.0 - float(count) / float(sumRequired)
        probs[lang] = prob
    #print("   probs", probs)

    links = None
    rnd = np.random.rand(1)
    #print("rnd", rnd, len(probs))
    cumm = 0.0
    for lang, prob in probs.items():
        cumm += prob
        #print("prob", prob, cumm)
        if cumm > rnd[0]:
            if lang in langsTodo:
                links = langsTodo[lang]
            break
    
    if links is not None and len(links) > 0:
        link = links.pop(0)
    else:
        link = RandomLink(langsTodo)
    #print("   node", node.Debug())
    return link

def RandomLink(langsTodo):
    while True:
        idx = np.random.randint(0, len(langsTodo))
        langs = list(langsTodo.keys())
        lang = langs[idx]
        links = langsTodo[lang]
        #print("idx", idx, len(nodes))
        if len(links) > 0:
            return links.pop(0)
    raise Exception("shouldn't be here")

def AddTodo(langsTodo, visited, link):
    childNode = link.childNode
    
    if childNode.urlId in visited:
        return

    parentNode = link.parentNode
    parentLang = parentNode.lang

    if parentLang not in langsTodo:
        langsTodo[parentLang] = []
    langsTodo[parentLang].append(link)

######################################################################################
######################################################################################
class Corpus:
    def __init__(self, params):
        self.params = params
        self.transitions = []
        self.losses = []
        self.sumWeights = []

    def AddTransition(self, transition):
        if self.params.deleteDuplicateTransitions:
            for currTrans in self.transitions:
                if currTrans.currURLId == transition.currURLId and currTrans.nextURLId == transition.nextURLId:
                    return
            # completely new trans
    
        self.transitions.append(transition)

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
class PolicyNetwork(nn.Module):
    def __init__(self, params, env):
        super(PolicyNetwork, self).__init__()
        self.NUM_ACTIONS = env.maxLangId + 1
        NUM_FEATURES = (env.maxLangId + 1) * 2
        HIDDEN_DIM = 512
        
        self.corpus = Corpus(params)

        self.linear1 = nn.Linear(NUM_FEATURES, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear3 = nn.Linear(HIDDEN_DIM, self.NUM_ACTIONS)
        self.optimizer = optim.Adam(self.parameters(), lr=params.lrn_rate)

    def forward(self, langsVisited, candidateCounts):
        #print("langsVisited", langsVisited)
        #print("candidateCounts", candidateCounts)
        x = torch.cat((langsVisited, candidateCounts), 1)
        #print("x", x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x 
    
    def get_action(self, langsVisited, candidateCounts):
        #print("langsVisited", type(langsVisited), langsVisited.shape, langsVisited)
        langsVisited = torch.from_numpy(langsVisited).float().unsqueeze(0)
        #print("state", type(state), state.shape)
        langsVisited = Variable(langsVisited)
        #print("   state", type(state), state.shape, state)

        candidateCounts = torch.from_numpy(candidateCounts).float().unsqueeze(0)
        candidateCounts = Variable(candidateCounts)

        probs = self.forward(langsVisited, candidateCounts)
        #print("probs", type(probs), probs.shape, probs)

        highest_prob_action = np.random.choice(self.NUM_ACTIONS, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        #print("probs", highest_prob_action, probs)

        return highest_prob_action, log_prob, probs

######################################################################################
def GetNextState(env, params, action, visited, candidates):
    #print("action", action)
    #print("candidates", candidates.Debug())
    #if action == 0:
    #    stopNode = env.nodes[0]
    #    link = Link("", 0, stopNode, stopNode)
    #elif not candidates.HasLinks(action):
    if action == 0 or not candidates.HasLinks(action):
        numLinks = candidates.CountLinks()
        #print("numLinks", numLinks)
        if numLinks > 0:
            link = candidates.RandomLink()
            #print("link1", link.childNode.Debug())
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

def NeuralWalk(env, params, eps, candidates, visited, langsVisited, qNet):
    langsVisited = np.squeeze(langsVisited, (0,))
    #print("langsVisited", langsVisited.shape, langsVisited)
    candidateCounts = candidates.GetCounts()
    #print("candidateCounts", candidateCounts.shape, candidateCounts)
    
    action, logProb, probs = qNet.get_action(langsVisited, candidateCounts)
    #print("action", action, logProb)

    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)
    #print("action", action, qValues, link, reward)

    return action, logProb, link, reward, probs

######################################################################################
def Trajectory(env, epoch, params, pNet):
    actions = []
    log_probs = []
    rewards = []
    visited = set()
    langsVisited = np.zeros([1, env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    while True:
        #print("visited", visited, node.urlId)
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        action, logProb, link, reward, probs = NeuralWalk(env, params, params.eps, candidates, visited, langsVisited, pNet)
        node = link.childNode
        actions.append(action)
        log_probs.append(logProb)
        rewards.append(reward)
        
        if link.childNode.urlId == 0:
            break

        if len(visited) > params.maxDocs:
            break

    #print("actions", actions)
    #print()

    return actions, log_probs, rewards

######################################################################################
def Walk(env, params, pNet):
    ret = []
    actions = []
    log_probs = []
    rewards = []

    visited = set()
    langsVisited = np.zeros([1, env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    mainStr = "nodes:" + str(node.urlId)
    rewardStr = "rewards:"

    i = 0
    numAligned = 0
    totReward = 0.0
    totDiscountedReward = 0.0
    discount = 1.0

    while True:
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        action, logProb, link, reward, probs = NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, pNet)
        node = link.childNode
        #print("action", action)
        actions.append(action)
        log_probs.append(logProb)
        rewards.append(reward)

        totReward += reward
        totDiscountedReward += discount * reward

        mainStr += "->" + str(node.urlId)
        rewardStr += "->" + str(reward)

        if node.alignedNode is not None:
            mainStr += "*"
            numAligned += 1

        discount *= params.gamma
        i += 1

        if node.urlId == 0:
            break

        if len(visited) > params.maxDocs:
            break

    mainStr += " " + str(i) 
    rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

    print("actions", actions)
    print(mainStr)
    print(rewardStr)
    return ret

######################################################################################
def Update(gamma, policy_network, log_probs, rewards):
    #print("log_probs", log_probs, rewards)
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    #print("discounted_rewards", discounted_rewards)    
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(unbiased=False) + 1e-9) # normalize discounted rewards
    #print("   rewards", len(rewards), type(discounted_rewards), discounted_rewards.shape)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    #print("policy_gradient", len(policy_gradient), policy_gradient)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    #print("policy_gradient", type(policy_gradient), policy_gradient.shape, policy_gradient)

    policy_gradient.backward()
    policy_network.optimizer.step()

def Train(params, saver, env, pNet):
    totRewards = []
    totDiscountedRewards = []

    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        TIMER.Start("Trajectory")
        actions, log_probs, rewards = Trajectory(env, epoch, params, pNet)
        #print("actions", actions, log_probs, rewards)
        TIMER.Pause("Trajectory")

        TIMER.Start("Update")
        Update(params.gamma, pNet, log_probs, rewards)
        TIMER.Pause("Update")

        if epoch > 0 and epoch % params.walk == 0:
            arrNaive = naive(env, len(env.nodes), params)
            arrBalanced = balanced(env, len(env.nodes), params)
            arrRL = Walk(env, params, pNet)
            print("epoch", epoch)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(arrNaive, label="naive")
            ax.plot(arrBalanced, label="balanced")
            ax.plot(arrRL, label="RL")
            ax.legend(loc='upper left')
            fig.show()
            plt.pause(0.001)


    return totRewards, totDiscountedRewards

######################################################################################
def main():
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

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    sqlconn = MySQL(options.configFile)

    languages = Languages(sqlconn.mycursor)
    params = LearningParams(languages, options.saveDir, options.deleteDuplicateTransitions, options.langPair)

    #hostName = "http://vade-retro.fr/"
    hostName = "http://www.buchmann.ch/"
    #hostName = "http://www.visitbritain.com/"
    env = Env(sqlconn, hostName)

    # change language of start node. 0 = stop
    env.nodes[sys.maxsize].lang = languages.GetLang("None")
    #for node in env.nodes.values():
    #    print(node.Debug())

    pNet = PolicyNetwork(params, env)

    saver = None #tf.train.Saver()

    totRewards, totDiscountedRewards = Train(params, saver, env, pNet)

    #params.debug = True
    arrNaive = naive(env, len(env.nodes), params)
    arrBalanced = balanced(env, len(env.nodes), params)
    arrRL = Walk(env, params, pNet)
    #print("arrNaive", arrNaive)
    #print("arrBalanced", arrBalanced)
    
    plt.plot(arrNaive, label="naive")
    plt.plot(arrBalanced, label="balanced")
    plt.plot(arrRL, label="RL")
    plt.legend(loc='upper left')
    plt.show()

######################################################################################
main()
