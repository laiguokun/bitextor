#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import tensorflow as tf
from tldextract import extract

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab as plt

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from common import GetLanguages, Languages, Timer
from helpers import GetEnvs, GetVistedSiblings, GetMatchedSiblings, NumParallelDocs, Env, Link
from corpus import Corpus
from neural_net import Qnets, Qnetwork
from save_plot import SavePlots
from candidate import Candidates

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, saveDirPlots, deleteDuplicateTransitions, langPair, maxLangId, defaultLang):
        self.gamma = 0.999
        self.lrn_rate = 0.001
        self.alpha = 0.7
        self.max_epochs = 100001
        self.eps = 0.1
        self.maxBatchSize = 1
        self.minCorpusSize = 200
        self.overSampling = 1
        
        self.debug = False
        self.walk = 10
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.saveDirPlots = saveDirPlots

        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 100.0 #17.0
        self.cost = -1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = 9999999999

        self.maxLangId = maxLangId
        self.defaultLang = defaultLang
        self.MAX_NODES = 1000

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = np.empty([1,2], dtype=np.int32)
        self.langIds[0,0] = languages.GetLang(langPairList[0])
        self.langIds[0,1] = languages.GetLang(langPairList[1])
        #print("self.langs", self.langs)

######################################################################################
######################################################################################
class Transition:
    def __init__(self, currURLId, nextURLId, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.numActions = numActions
        self.linkLang = np.array(linkLang, copy=True) 
        self.mask = np.array(mask, copy=True) 
        self.numSiblings = np.array(numSiblings, copy=True) 
        self.numVisitedSiblings = np.array(numVisitedSiblings, copy=True) 
        self.numMatchedSiblings = np.array(numMatchedSiblings, copy=True) 
        self.langIds = langIds 
        self.langsVisited = np.array(langsVisited, copy=True)
        self.targetQ = np.array(targetQ, copy=True)

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret
    
######################################################################################
def GetNextState(env, params, action, visited, candidates, linkLang, numSiblings, numVisitedSiblings, numMatchedSiblings):
    #print("candidates", action, candidates.Debug())
    if action == -1:
        # no explicit stop state but no candidates
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
    else:
        langId = linkLang[0, action]
        numSiblings1 = numSiblings[0, action]
        numVisitedSiblings1 = numVisitedSiblings[0, action]
        numMatchedSiblings1 = numMatchedSiblings[0, action]
        key = (langId, numSiblings1, numVisitedSiblings1, numMatchedSiblings1)
        link = candidates.Pop(key)
 
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
def NeuralWalk(env, params, eps, candidates, visited, langsVisited, sess, qnA):
    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action = qnA.PredictAll(env, sess, params.langIds, langsVisited, candidates)
    #print("action", action, linkLang, qValues)
    if action >= 0:
        if np.random.rand(1) < eps:
            #print("actions", type(actions), actions)
            action = np.random.randint(0, numActions)
            maxQ = qValues[0, action]
            #print("random")
        #print("action", action, qValues)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates, linkLang, numSiblings, numVisitedSiblings, numMatchedSiblings)
    assert(link is not None)
    #print("action", action, qValues, link.childNode.Debug(), reward)

    return numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action, link, reward

######################################################################################
def Neural(env, params, candidates, visited, langsVisited, sess, qnA, qnB):
    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action, link, reward = NeuralWalk(env, params, params.eps, candidates, visited, langsVisited, sess, qnA)
    assert(link is not None)
    
    # calc nextMaxQ
    nextVisited = visited.copy()
    nextVisited.add(link.childNode.urlId)

    nextCandidates = candidates.copy()
    nextCandidates.AddLinks(link.childNode, nextVisited, params)

    nextLangsVisited = langsVisited.copy()
    UpdateLangsVisited(nextLangsVisited, link.childNode, params.langIds)

    if nextCandidates.Count() > 0:
        _, _, _, _, _, _, _, _, nextAction = qnA.PredictAll(env, sess, params.langIds, nextLangsVisited, nextCandidates)
        #print("nextAction", nextAction, nextLangRequested, nextCandidates.Debug())
        _, _, _, _, _, _, nextQValuesB, _, _ = qnB.PredictAll(env, sess, params.langIds, nextLangsVisited, nextCandidates)
        nextMaxQ = nextQValuesB[0, nextAction]
        #print("nextMaxQ", nextMaxQ, nextMaxQB, nextQValuesA[0, nextAction])
    else:
        nextMaxQ = 0

    newVal = reward + params.gamma * nextMaxQ
    targetQ = (1 - params.alpha) * maxQ + params.alpha * newVal
    qValues[0, action] = targetQ

    transition = Transition(link.parentNode.urlId, 
                            link.childNode.urlId,
                            numActions,
                            linkLang,
                            mask,
                            numSiblings,
                            numVisitedSiblings,
                            numMatchedSiblings,
                            params.langIds,
                            langsVisited,
                            qValues)

    return transition

######################################################################################
def UpdateLangsVisited(langsVisited, node, langIds):
        if node.lang == langIds[0, 0]:
            langsVisited[0, 0] += 1
        elif node.lang == langIds[0, 1]:
            langsVisited[0, 1] += 1
        else:
            langsVisited[0, 2] += 1

######################################################################################
def Trajectory(env, epoch, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, 3]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    while True:
        tmp = np.random.rand(1)
        if tmp > 0.5:
            qnA = qns.q[0]
            qnB = qns.q[1]
        else:
            qnA = qns.q[1]
            qnB = qns.q[0]

        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)

        UpdateLangsVisited(langsVisited, node, params.langIds)        
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        transition = Neural(env, params, candidates, visited, langsVisited, sess, qnA, qnB)

        if transition.nextURLId == 0:
            break
        else:
            tmp = np.random.rand(1)
            if tmp > 0.5:
                corpus = qnA.corpus
            else:
                corpus = qnB.corpus

            corpus.AddTransition(transition)
            node = env.nodes[transition.nextURLId]

        if len(visited) > params.maxDocs:
            break

    return ret

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, 3]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    mainStr = "lang:" + str(node.lang)
    rewardStr = "rewards:"
    actionStr = "actions:"

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
        #print("node.lang", node.lang, langsVisited.shape)
        UpdateLangsVisited(langsVisited, node, params.langIds)        
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        _, _, _, _, _, _, _, _, action, link, reward = NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, sess, qnA)
        node = link.childNode
        #print("action", action, qValues)
        actionStr += str(action) + " "

        totReward += reward
        totDiscountedReward += discount * reward

        mainStr += "->" + str(node.lang)
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

    print(actionStr)
    print(mainStr)
    print(rewardStr)
    return ret, totReward, totDiscountedReward

######################################################################################
def Train(params, sess, saver, qns, envs, envsTest):
    print("Start training")
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        for env in envs:
            TIMER.Start("Trajectory")
            _ = Trajectory(env, epoch, params, sess, qns)
            TIMER.Pause("Trajectory")

        TIMER.Start("Train")
        qns.q[0].corpus.Train(sess, params)
        qns.q[1].corpus.Train(sess, params)
        TIMER.Pause("Train")

        if epoch > 0 and epoch % params.walk == 0:
            print("epoch", epoch)
            SavePlots(sess, qns, params, envs, params.saveDirPlots, epoch, "train")
            SavePlots(sess, qns, params, envsTest, params.saveDirPlots, epoch, "test")

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
    oparser.add_argument("--save-plots", dest="saveDirPlots", default="plot",
                     help="Directory ")
    oparser.add_argument("--delete-duplicate-transitions", dest="deleteDuplicateTransitions",
                         default=False, help="If True then only unique transition are used in each batch")
    oparser.add_argument("--num-train-hosts", dest="numTrainHosts", type=int,
                         default=1, help="Number of domains to train on")
    oparser.add_argument("--num-test-hosts", dest="numTestHosts", type=int,
                         default=3, help="Number of domains to test on")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    languages = GetLanguages(options.configFile)
    params = LearningParams(languages, options.saveDir, options.saveDirPlots, options.deleteDuplicateTransitions, options.langPair, languages.maxLangId, languages.GetLang("None"))

    if not os.path.exists(options.saveDirPlots): os.mkdir(options.saveDirPlots)

    print("options.numTrainHosts", options.numTrainHosts)
    #hosts = ["http://vade-retro.fr/"]
    hosts = ["http://www.buchmann.ch/", "http://telasmos.org/", "http://tagar.es/"]
    #hosts = ["http://www.visitbritain.com/"]

    #hostsTest = ["http://vade-retro.fr/"]
    #hostsTest = ["http://www.visitbritain.com/"]
    hostsTest = ["http://www.visitbritain.com/", "http://chopescollection.be/", "http://www.bedandbreakfast.eu/"]

    envs = GetEnvs(options.configFile, languages, hosts[:options.numTrainHosts])
    envsTest = GetEnvs(options.configFile, languages, hostsTest[:options.numTestHosts])

    tf.reset_default_graph()
    qns = Qnets(params)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        Train(params, sess, saver, qns, envs, envsTest)

######################################################################################
main()
