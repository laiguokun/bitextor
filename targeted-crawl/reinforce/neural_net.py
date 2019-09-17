import os
import sys
import numpy as np
import tensorflow as tf

from corpus import Corpus

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from helpers import Link
from candidate import GetLangsVisited

######################################################################################
def GetNextState(env, params, action, visited, candidates):

    #print("candidates", action, candidates.Debug())
    if action == -1:
        # no explicit stop state but no candidates
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
    else:
        _, linkLang, _, numSiblings, numVisitedSiblings, numMatchedSiblings = candidates.GetFeatures()
        langId = linkLang[0, action]
        numSiblings1 = numSiblings[0, action]
        numVisitedSiblings1 = numVisitedSiblings[0, action]
        numMatchedSiblings1 = numMatchedSiblings[0, action]
        key = (langId, numSiblings1, numVisitedSiblings1, numMatchedSiblings1)
        link = candidates.Pop(key)
        candidates.AddLinks(link.childNode, visited, params)

    assert(link.childNode.urlId not in visited)
    visited.add(link.childNode.urlId)
 
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
def NeuralWalk(env, params, eps, candidates, visited, sess, qnA):
    qValues, maxQ, action = qnA.PredictAll(env, sess, params.langIds, visited, candidates)

    #print("action", action, linkLang, qValues)
    if action >= 0:
        if np.random.rand(1) < eps:
            #print("actions", type(actions), actions)
            numActions, _, _, _, _, _ = candidates.GetFeatures()
            action = np.random.randint(0, numActions)
            maxQ = qValues[0, action]
            #print("random")
        #print("action", action, qValues)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)
    #print("action", action, qValues, link.childNode.Debug(), reward)

    return qValues, maxQ, action, link, reward

######################################################################################
class Qnetwork():
    def __init__(self, params):
        self.params = params
        self.corpus = Corpus(params, self)

        HIDDEN_DIM = 512

        # mask
        self.mask = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.bool)
        self.maskInt8 = tf.cast(self.mask, dtype=tf.int8)
        self.maskInt8Neg = tf.multiply(tf.add(self.maskInt8, -1), -1)

        # graph represention
        self.langIds = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.langsVisited = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # link representation
        self.linkLang = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numVisitedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numMatchedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)

        # batch size
        self.batchSize = tf.shape(self.linkLang)[0]
        
        # network
        self.input = tf.concat([self.langIds, self.langsVisited, self.numActions], 1)
        #print("self.input", self.input.shape)

        self.W1 = tf.Variable(tf.random_uniform([3 + 3, HIDDEN_DIM], 0, 0.01))
        self.b1 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.input, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        #self.hidden1 = tf.nn.relu(self.hidden1)
        self.hidden1 = tf.math.sigmoid(self.hidden1)
        #print("self.hidden1", self.hidden1.shape)

        self.W2 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b2 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        #self.hidden2 = tf.nn.relu(self.hidden2)
        self.hidden2 = tf.math.sigmoid(self.hidden2)
        #print("self.hidden2", self.hidden2.shape)

        self.W3 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b3 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden3 = tf.matmul(self.hidden2, self.W3)
        self.hidden3 = tf.add(self.hidden3, self.b3)
        #self.hidden3 = tf.nn.relu(self.hidden3)
        self.hidden3 = tf.math.sigmoid(self.hidden3)

        # link-specific
        self.WlinkSpecific = tf.Variable(tf.random_uniform([4, HIDDEN_DIM], 0, 0.01))
        self.blinkSpecific = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.linkSpecific = tf.stack([tf.transpose(self.linkLang), 
                                    tf.transpose(self.numSiblings), 
                                    tf.transpose(self.numVisitedSiblings),
                                    tf.transpose(self.numMatchedSiblings)], 0)
        self.linkSpecific = tf.transpose(self.linkSpecific)
        self.linkSpecific = tf.reshape(self.linkSpecific, [self.batchSize * self.params.MAX_NODES, 4])

        self.linkSpecific = tf.matmul(self.linkSpecific, self.WlinkSpecific)
        self.linkSpecific = tf.add(self.linkSpecific, self.blinkSpecific)        
        self.linkSpecific = tf.nn.relu(self.linkSpecific)
        #self.linkSpecific = tf.nn.sigmoid(self.linkSpecific)
        self.linkSpecific = tf.reshape(self.linkSpecific, [self.batchSize, self.params.MAX_NODES, 512])

        # final q-values
        self.hidden3 = tf.reshape(self.hidden3, [self.batchSize, 1, HIDDEN_DIM])
        self.hidden3 = tf.multiply(self.linkSpecific, self.hidden3)
        self.hidden3 = tf.reduce_sum(self.hidden3, axis=2)

        #self.qValues = self.hidden3
        self.qValues = tf.boolean_mask(self.hidden3, self.mask, axis=0)

        self.maxQ = tf.multiply(self.hidden3, tf.cast(self.maskInt8, dtype=tf.float32))

        # softmax
        self.probs = tf.nn.softmax(self.qValues, axis=0)
        self.chosenAction = tf.argmax(self.probs,0)

        # REINFORCE
        self.reward_holder = tf.placeholder(shape=[None, 1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None, 1],dtype=tf.int32) #  0 or 1
       
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        nextQMasked = tf.boolean_mask(self.nextQ, self.mask, axis=0)

        self.loss = nextQMasked - self.qValues
        #self.loss = tf.reduce_max(tf.square(self.loss))
        self.loss = tf.reduce_mean(tf.square(self.loss))
        #self.loss = tf.reduce_sum(tf.square(self.loss))
        
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer(learning_rate=params.lrn_rate)
        
        self.updateModel = self.trainer.minimize(self.loss)

        #self.sumWeight = tf.reduce_sum(self.W1) \
        #                 + tf.reduce_sum(self.b1) \
        #                 + tf.reduce_sum(self.W2) \
        #                 + tf.reduce_sum(self.b2) \
        #                 + tf.reduce_sum(self.W3) \
        #                 + tf.reduce_sum(self.b3) 

    def PredictAll(self, env, sess, langIds, visited, candidates):
        numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings = candidates.GetFeatures()
        
        numActionsNP = np.empty([1,1], dtype=np.int32)
        numActionsNP[0,0] = numActions

        assert(numActions > 0)

        #print("linkLang", numActions, linkLang.shape)
        #print("mask", mask.shape, mask)
        langsVisited = GetLangsVisited(visited, langIds, env)
        #print("langsVisited", langsVisited)
        
        (qValues, probs, chosenAction, maxQ) = sess.run([self.qValues, self.probs, self.chosenAction, self.maxQ], 
                                feed_dict={self.linkLang: linkLang,
                                    self.numActions: numActionsNP,
                                    self.mask: mask,
                                    self.numSiblings: numSiblings,
                                    self.numVisitedSiblings: numVisitedSiblings,
                                    self.numMatchedSiblings: numMatchedSiblings,
                                    self.langIds: langIds,
                                    self.langsVisited: langsVisited})
        #print("hidden3", hidden3.shape, hidden3)
        #print("qValues", qValues.shape, qValues)
        #print("   maxQ", maxQ.shape, maxQ)
        #print("  probs", probs.shape, probs)
        #print("  chosenAction", chosenAction.shape, chosenAction)
        #print("linkSpecific", linkSpecific.shape)
        #print("numSiblings", numSiblings.shape)
        #print("numVisitedSiblings", numVisitedSiblings.shape)
        #print("numMatchedSiblings", numMatchedSiblings.shape)
        qValues = np.reshape(qValues, [1, qValues.shape[0] ])
        #print("   qValues", qValues)
        #print()

        action = np.random.choice(probs,p=probs)
        #print("  action", action)
        action = np.argmax(probs == action)
        #print("  action", action)

        maxQ = qValues[0, action]
        #print("newAction", action, maxQ)

        return qValues, maxQ, action

    def Update(self, sess, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ, actions, discountedRewards):
        #print("actions, discountedRewards", actions, discountedRewards)
        #print("input", linkLang.shape, langIds.shape, langFeatures.shape, targetQ.shape)
        #print("targetQ", targetQ)
        _, loss, hidden3, qValues, maxQ, maskInt8, maskInt8Neg = sess.run([self.updateModel, self.loss, self.hidden3, self.qValues, self.maxQ, self.maskInt8, self.maskInt8Neg], 
                                    feed_dict={self.linkLang: linkLang, 
                                            self.numActions: numActions,
                                            self.mask: mask,
                                            self.numSiblings: numSiblings,
                                            self.numVisitedSiblings: numVisitedSiblings,
                                            self.numMatchedSiblings: numMatchedSiblings,
                                            self.langIds: langIds, 
                                            self.langsVisited: langsVisited,
                                            self.nextQ: targetQ,
                                            self.action_holder: actions,
                                            self.reward_holder: discountedRewards})
        #print("loss", loss, numActions)
        print("hidden3", hidden3.shape, hidden3)
        print("   qValues", qValues.shape, qValues)
        #print("   maskInt8", maskInt8.shape, maskInt8)
        print("   maskInt8Neg", maskInt8Neg.shape, maskInt8Neg)
        print("   maxQ", maxQ.shape, maxQ)
        return loss

