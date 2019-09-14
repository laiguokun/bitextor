#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import tensorflow as tf

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from common import GetLanguages, Languages, Timer
from helpers import GetEnvs, GetVistedSiblings, GetMatchedSiblings, NumParallelDocs, Env, Link
from corpus import Corpus
from neural_net import Qnets, Qnetwork, NeuralWalk, GetNextState
from save_plot import SavePlots, Walk
from candidate import Candidates, GetLangsVisited

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
def GetStartTransition(env, params, visited, candidates):
    node = env.nodes[sys.maxsize]
    
    assert(node.urlId not in visited)
    #print("node", node.Debug())
    visited.add(node.urlId)
    candidates.AddLinks(node, visited, params)

    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings = candidates.GetFeatures()
    transition = Transition(env, None, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, params.langIds, 0, visited, candidates)
    return transition

######################################################################################
class Transition:
    def __init__(self, env, link, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, targetQ, visited, candidates):
        self.link = link
        self.numActions = numActions
        self.linkLang = np.array(linkLang, copy=True) 
        self.mask = np.array(mask, copy=True) 
        self.numSiblings = np.array(numSiblings, copy=True) 
        self.numVisitedSiblings = np.array(numVisitedSiblings, copy=True) 
        self.numMatchedSiblings = np.array(numMatchedSiblings, copy=True) 
        self.langIds = langIds 
        self.targetQ = np.array(targetQ, copy=True)
        self.candidates = candidates

        self.visited = visited.copy()
        langsVisited = GetLangsVisited(visited, langIds, env)
        self.langsVisited = np.array(langsVisited, copy=True)

    def Debug(self):
        ret = str(self.link.parentNode.urlId) + "->" + str(self.link.childNode.urlId) + " " + str(self.visited)
        return ret
    
######################################################################################
def Neural(env, params, prevTransition, sess, qnA, qnB):
    candidates = prevTransition.candidates
    visited = prevTransition.visited

    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings = candidates.GetFeatures()
    qValues, maxQ, action, link, reward = NeuralWalk(env, params, params.eps, candidates, visited, sess, qnA)
    assert(link is not None)
    
    # calc nextMaxQ
    nextVisited = visited.copy()
    nextVisited.add(link.childNode.urlId)

    nextCandidates = candidates.copy()
    nextCandidates.AddLinks(link.childNode, nextVisited, params)

    if nextCandidates.Count() > 0:
        _, _, nextAction = qnA.PredictAll(env, sess, params.langIds, nextVisited, nextCandidates)
        #print("nextAction", nextAction, nextLangRequested, nextCandidates.Debug())
        nextQValuesB, _, _ = qnB.PredictAll(env, sess, params.langIds, nextVisited, nextCandidates)
        nextMaxQ = nextQValuesB[0, nextAction]
        #print("nextMaxQ", nextMaxQ, nextMaxQB, nextQValuesA[0, nextAction])
    else:
        nextMaxQ = 0

    newVal = reward + params.gamma * nextMaxQ
    targetQ = (1 - params.alpha) * maxQ + params.alpha * newVal
    qValues[0, action] = targetQ

    transition = Transition(env, 
                            link,
                            numActions,
                            linkLang,
                            mask,
                            numSiblings,
                            numVisitedSiblings,
                            numMatchedSiblings,
                            params.langIds,
                            qValues,
                            visited,
                            candidates)

    return transition

######################################################################################
def Trajectory(env, epoch, params, sess, qns):
    ret = []
    visited = set()
    candidates = Candidates(params, env)
    transition = GetStartTransition(env, params, visited, candidates)

    while True:
        tmp = np.random.rand(1)
        if tmp > 0.5:
            qnA = qns.q[0]
            qnB = qns.q[1]
        else:
            qnA = qns.q[1]
            qnB = qns.q[0]

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        transition = Neural(env, params, transition, sess, qnA, qnB)
        print("visited", visited)
        print("transition", transition.Debug())

        if transition.link.childNode.urlId == 0:
            break
        else:
            tmp = np.random.rand(1)
            if tmp > 0.5:
                corpus = qnA.corpus
            else:
                corpus = qnB.corpus

            corpus.AddTransition(transition)
            node = transition.link.childNode

        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        #print("visited", visited)

        candidates.AddLinks(node, visited, params)

        if len(visited) > params.maxDocs:
            break

    return ret

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
