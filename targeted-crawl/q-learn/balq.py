#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import pylab as plt

from common import MySQL, Languages
from helpers import Env

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, deleteDuplicateTransitions, langPair):
        self.gamma = 0.99
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 20001
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
def naive(sqlconn, env, maxDocs, params):
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
def PopNode(langsTodo, langsVisited, params):
    sum = 0
    # any nodes left to do
    for nodes in langsTodo.values():
        sum += len(nodes)
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

    nodes = None
    rnd = np.random.rand(1)
    #print("rnd", rnd, len(probs))
    cumm = 0.0
    for lang, prob in probs.items():
        cumm += prob
        #print("prob", prob, cumm)
        if cumm > rnd[0]:
            if lang in langsTodo:
                nodes = langsTodo[lang]
            break
    
    if nodes is not None and len(nodes) > 0:
        node = nodes.pop(0)
    else:
        node = RandomNode(langsTodo)
    #print("   node", node.Debug())
    return node

def RandomNode(langsTodo):
    while True:
        idx = np.random.randint(0, len(langsTodo))
        langs = list(langsTodo.keys())
        lang = langs[idx]
        nodes = langsTodo[lang]
        #print("idx", idx, len(nodes))
        if len(nodes) > 0:
            return nodes.pop(0)
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
def balanced(sqlconn, env, maxDocs, params):
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

        link = PopNode(langsTodo, langsVisited, params)

    return ret

######################################################################################
def main():
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
        
    #params.debug = True
    arrNaive = naive(sqlconn, env, len(env.nodes), params)
    arrBalanced = balanced(sqlconn, env, len(env.nodes), params)
    #print("arrNaive", arrNaive)
    #print("arrBalanced", arrBalanced)
    
    plt.plot(arrNaive)
    plt.plot(arrBalanced)
    plt.show()

######################################################################################
main()
