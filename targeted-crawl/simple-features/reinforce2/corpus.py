import os
import sys
import numpy as np

######################################################################################
class Corpus:
    def __init__(self, params):
        self.params = params
        self.transitions = []
        self.losses = []

    def AddTransition(self, transition):    
        self.transitions.append(transition)

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def CalcDiscountedReward(self):
        runningReward = 0.0
        for t in reversed(range(0, len(self.transitions))):
            transition = self.transitions[t]
            runningReward = runningReward * self.params.gamma + transition.reward
            transition.discountedReward = runningReward
            #print("t", t, transition.Debug())

    def CalcGrads(self, sess, qn):
        self.CalcDiscountedReward()

        #for transition in self.transitions:
        #    print(transition.Debug())
        #lastTrans = self.transitions[-1]
        #print("lastTrans", lastTrans.Debug())

        #numIter = len(self.transitions) * params.overSampling / params.maxBatchSize
        #numIter = int(numIter) + 1
        #print("numIter", numIter, len(self.transitions), params.overSampling, params.maxBatchSize)
        #for i in range(numIter):
        #    batch = self.GetBatchWithoutDelete(params.maxBatchSize)
        #    loss = self.UpdateQN(params, sess, batch)
        #    self.losses.append(loss)

        batchSize = len(self.transitions)
        #print("batchSize", batchSize)
        numActions = np.empty([batchSize, 1], dtype=np.int)
        mask = np.empty([batchSize, self.params.MAX_NODES], dtype=np.bool)

        langIds = np.empty([batchSize, 2], dtype=np.int)
        langsVisited = np.empty([batchSize, 3])

        actions = np.empty([batchSize], dtype=np.int)
        discountedRewards = np.empty([batchSize], dtype=np.float32)
        
        i = 0
        for transition in self.transitions:
            #curr = transition.curr
            #next = transition.next
            #print("transition.numActions", transition.numActions, transition.targetQ.shape, transition.candidates.Count())
            numActions[i, 0] = transition.numActions
            mask[i, :] = transition.mask

            langIds[i, :] = transition.langIds
            langsVisited[i, :] = transition.langsVisited

            actions[i] = transition.action
            discountedRewards[i] = transition.discountedReward

            i += 1

        loss = qn.CalcGrads(sess, numActions, mask, langIds, langsVisited, actions, discountedRewards)

        self.transitions.clear()

        #print("loss", loss)
        return loss

