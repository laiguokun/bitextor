#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import pylab as plt
import tensorflow as tf

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
        self.gamma = 1.0 #0.9
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 2001
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
        self.maxDocs = 300 #9999999999

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
    def __init__(self, params, qn):
        self.params = params
        self.qn = qn
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

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def Train(self, sess, env, params):
        if len(self.transitions) >= params.minCorpusSize:
            #for transition in self.transitions:
            #    print(DebugTransition(transition))

            for i in range(params.trainNumIter):
                batch = self.GetBatchWithoutDelete(params.maxBatchSize)
                loss, sumWeight = self.UpdateQN(params, env, sess, batch)
                self.losses.append(loss)
                self.sumWeights.append(sumWeight)
            self.transitions.clear()
        
    def UpdateQN(self, params, env, sess, batch):
        batchSize = len(batch)
        #print("batchSize", batchSize)
        langRequested = np.empty([batchSize, 1], dtype=np.int)
        langIds = np.empty([batchSize, 2], dtype=np.int)
        langFeatures = np.empty([batchSize, env.maxLangId + 1])
        targetQ = np.empty([batchSize, 1])

        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next

            langRequested[i, :] = transition.langRequested
            langIds[i, :] = transition.langIds
            langFeatures[i, :] = transition.langFeatures
            targetQ[i, :] = transition.targetQ

            i += 1

        #_, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
        TIMER.Start("UpdateQN.1")
        loss, sumWeight = self.qn.Update(sess, langRequested, langIds, langFeatures, targetQ)
        TIMER.Pause("UpdateQN.1")

        #print("loss", loss)
        return loss, sumWeight

######################################################################################
class Transition:
    def __init__(self, currURLId, nextURLId, langRequested, langIds, langFeatures, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.langRequested = langRequested 
        self.langIds = langIds 
        self.langFeatures = langFeatures 
        self.targetQ = targetQ 

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

    def Debug(self):
        ret = ""
        for lang in self.dict:
            ret += "lang=" + str(lang) + ":"
            links = self.dict[lang]
            for link in links:
                ret += " " + link.parentNode.url + "->" + link.childNode.url
        return ret
    
######################################################################################
class Qnets():
    def __init__(self, params, env):
        HIDDEN_DIM = 512
        NUM_FEATURES = env.maxLangId + 1
        self.pq = PolicyNetwork(NUM_FEATURES, NUM_FEATURES, 512)

######################################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        print("num_inputs", num_inputs, num_actions, hidden_size, learning_rate)

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x 
    
    def get_action(self, state):
        print("   state", type(state), state.shape, state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        #print("state", type(state), state.shape)
        state = Variable(state)
        #print("   state", type(state), state.shape, state)
        probs = self.forward(state)
        print("   probs", type(probs), probs.shape, probs)

        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        #print("probs", type(probs), probs.shape, probs, highest_prob_action, log_prob)
        return highest_prob_action, log_prob

######################################################################################
def GetNextState(env, params, action, visited, candidates):
    if action == 0:
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
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

def NeuralWalk(env, params, eps, candidates, visited, langsVisited, sess, qp):
    langsVisited = np.squeeze(langsVisited, (0,))
    #print("langsVisited", langsVisited.shape, langsVisited)
    action, logProb = qp.get_action(langsVisited)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)
    #print("action", action, qValues, link, reward)

    return action, logProb, link, reward

def Neural(env, params, candidates, visited, langsVisited, sess, qp):
    action, logProb,link, reward = NeuralWalk(env, params, params.eps, candidates, visited, langsVisited, sess, qp)
    assert(link is not None)
    #print("action", action, qValues, link, reward)
    
    transition = Transition(link.parentNode.urlId, 
                            link.childNode.urlId,
                            action,
                            params.langIds,
                            langsVisited,
                            targetQ)

    return transition

######################################################################################
def Trajectory(env, epoch, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    while True:
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        transition = Neural(env, params, candidates, visited, langsVisited, sess, qns.pq)

        if transition.nextURLId == 0:
            break
        else:
            qns.pq.corpus.AddTransition(transition)
            node = env.nodes[transition.nextURLId]

        if len(visited) > params.maxDocs:
            break

    return ret

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
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
        qnA = qns.q[0]
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        qValues, _, action, link, reward = NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, sess, qnA)
        node = link.childNode
        print("action", action, qValues)

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

    print(mainStr)
    print(rewardStr)
    return ret

######################################################################################
def Train(params, sess, saver, env, qns):
    totRewards = []
    totDiscountedRewards = []

    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        TIMER.Start("Trajectory")
        _ = Trajectory(env, epoch, params, sess, qns)

        TIMER.Pause("Trajectory")

        TIMER.Start("Update")
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)
        TIMER.Pause("Update")

        if epoch > 0 and epoch % params.walk == 0:
            #arrNaive = naive(env, len(env.nodes), params)
            #arrBalanced = balanced(env, len(env.nodes), params)
            _ = Walk(env, params, sess, qns)
            print("epoch", epoch)

            #plt.plot(arrNaive, label="naive")
            #plt.plot(arrBalanced, label="balanced")
            #plt.plot(arrRL, label="RL")
            #plt.legend(loc='upper left')
            #plt.show()


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

    hostName = "http://vade-retro.fr/"
    #hostName = "http://www.buchmann.ch/"
    #hostName = "http://www.visitbritain.com/"
    env = Env(sqlconn, hostName)

    # change language of start node. 0 = stop
    env.nodes[sys.maxsize].lang = languages.GetLang("None")
    #for node in env.nodes.values():
    #    print(node.Debug())

    tf.reset_default_graph()
    qns = Qnets(params, env)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        totRewards, totDiscountedRewards = Train(params, sess, saver, env, qns)

        #params.debug = True
        arrNaive = naive(env, len(env.nodes), params)
        arrBalanced = balanced(env, len(env.nodes), params)
        arrRL = Walk(env, params, sess, qns)
        #print("arrNaive", arrNaive)
        #print("arrBalanced", arrBalanced)
        
        plt.plot(arrNaive, label="naive")
        plt.plot(arrBalanced, label="balanced")
        plt.plot(arrRL, label="RL")
        plt.legend(loc='upper left')
        plt.show()

######################################################################################
main()
