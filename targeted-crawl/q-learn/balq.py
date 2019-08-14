#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import pylab as plt
import tensorflow as tf

from common import MySQL, Languages, Timer
from helpers import Env

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, deleteDuplicateTransitions, langPair):
        self.gamma = 0.99
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 2 #20001
        self.eps = 1 # 0.7
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 10
        
        self.debug = False
        self.walk = 1000
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 17.0
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
class Transition:
    def __init__(self, currURLId, nextURLId):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

class Candidates:
    def __init__(self, params):
        self.params = params
        self.dict = {} # parent lang -> links[]

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params)

        for key, value in self.dict.items():
            #print("key", key, value)
            ret.dict[key] = value.copy()

        return ret
    
    def AddLink(self, link):
        langId = link.parentNode.lang
        if langId not in self.dict:
            self.dict[langId] = []
        self.dict[langId].append(link)

    def GetLangProb(self, langRequested, langIds, langsVisited, params):
        # sum of all nodes visited
        sumRequired = 0
        for lang, count in langsVisited.items():
            if lang in langIds:
                sumRequired += count
        sumRequired += 0.001 #1
        #print("langsVisited", sumAll, sumRequired, langsVisited)
        
        if langRequested in langsVisited:
            ret = 1 - float(langsVisited[langRequested]) / float(sumRequired)
        else:
            ret = 0
        return ret

    def RandomLink(self):
        while True:
            idx = np.random.randint(0, len(self.dict))
            langs = list(self.dict.keys())
            lang = langs[idx]
            links = self.dict[lang]
            #print("idx", idx, len(nodes))
            if len(links) > 0:
                return links.pop(0)
        raise Exception("shouldn't be here")
    
######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        self.corpus = Corpus(params, self)

######################################################################################
class Qnets():
    def __init__(self, params, env):
        self.q = []
        self.q.append(Qnetwork(params, env))
        self.q.append(Qnetwork(params, env))

######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.params = params
        self.qn = qn
        self.transitions = []


    def AddTransition(self, transition):
        if self.params.deleteDuplicateTransitions:
            for currTrans in self.transitions:
                if currTrans.currURLId == transition.currURLId and currTrans.nextURLId == transition.nextURLId:
                    return
            # completely new trans
    
        self.transitions.append(transition)
            
    def Train(self, sess, env, params):
        pass

######################################################################################
def Neural(params, unvisited, langsVisited):
    #link = unvisited.Pop(langsVisited, params)
    sum = 0
    # any nodes left to do
    for nodes in unvisited.dict.values():
        sum += len(nodes)
    if sum == 0:
        return Transition(0, 0)
    del sum

    probs = {}
    for langId in params.langIds:
        prob = unvisited.GetLangProb(langId, params.langIds, langsVisited, params)
        probs[langId] = prob

    maxProb = 0
    argMax = sys.maxsize
    for langId in params.langIds:
        if probs[langId] > maxProb:
            maxProb = probs[langId]
            argMax = langId

    if argMax in unvisited.dict:
        links = unvisited.dict[argMax]
        if len(links) > 0:
            link = links.pop(0)
        else:
            link = unvisited.RandomLink()
    else:
        link = unvisited.RandomLink()

    if link is not None:
        transition = Transition(link.parentNode.urlId, link.childNode.urlId)
    else:
        transition = Transition(0, 0)

    return transition

######################################################################################
def Trajectory(env, epoch, params, qns):
    ret = []
    visited = set()
    langsVisited = {} # langId -> count
    candidates = Candidates(params)
    node = env.nodes[sys.maxsize]

    while True:
        tmp = np.random.rand(1)
        if tmp > 0.5:
            qnA = qns.q[0]
            qnB = qns.q[1]
        else:
            qnA = qns.q[1]
            qnB = qns.q[0]

        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)
    
            if len(visited) > params.maxDocs:
                break

            for link in node.links:
                #print("   ", childNode.Debug())
                candidates.AddLink(link)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

        transition = Neural(params, candidates, langsVisited)
        qnA.corpus.AddTransition(transition)

        if transition.nextURLId == 0:
            break
        else:
            node = env.nodes[transition.nextURLId]

    return ret

######################################################################################
def Train(params, sess, saver, env, qns):
    totRewards = []
    totDiscountedRewards = []

    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        #startState = 30
        
        TIMER.Start("Trajectory")
        arrRL = Trajectory(env, epoch, params, qns)

        TIMER.Pause("Trajectory")

        TIMER.Start("Update")
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)
        TIMER.Pause("Update")

        arrNaive = naive(env, len(env.nodes), params)
        arrBalanced = balanced(env, len(env.nodes), params)
        plt.plot(arrNaive, label="naive")
        plt.plot(arrBalanced, label="balanced")
        plt.plot(arrRL, label="RL")
        plt.legend(loc='upper left')
        plt.show()


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
    #hostName = "http://www.buchmann.ch/"
    hostName = "http://www.visitbritain.com/"
    env = Env(sqlconn, hostName)

    tf.reset_default_graph()
    qns = Qnets(params, env)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        totRewards, totDiscountedRewards = Train(params, sess, saver, env, qns)

        #params.debug = True
        #arrNaive = naive(env, len(env.nodes), params)
        #arrBalanced = balanced(env, len(env.nodes), params)
        #arrRL = Trajectory(env, len(env.nodes), params, qns)
        #print("arrNaive", arrNaive)
        #print("arrBalanced", arrBalanced)
        
        #plt.plot(arrNaive, label="naive")
        #plt.plot(arrBalanced, label="balanced")
        #plt.plot(arrRL, label="RL")
        #plt.legend(loc='upper left')
        #plt.show()

######################################################################################
main()
