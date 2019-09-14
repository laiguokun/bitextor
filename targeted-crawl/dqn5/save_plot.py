import os
import sys
import numpy as np
from tldextract import extract
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab as plt

from other_strategies import dumb, randomCrawl, balanced
from candidate import Candidates
from helpers import NumParallelDocs
from neural_net import NeuralWalk, GetNextState

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
    node = env.nodes[sys.maxsize]
    visited = set()
    visited.add(node.urlId)
    candidates = Candidates(params, env)
    candidates.AddLinks(node, visited, params)

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

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        _, _, action, link, reward = NeuralWalk(env, params, 0.0, candidates, visited, sess, qnA)
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


