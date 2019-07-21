#!/usr/bin/env python3

import sys
import os
import numpy as np
import pylab as plt
import tensorflow as tf
import random
from collections import namedtuple
import mysql.connector
import time
import argparse
import pickle

######################################################################################
class Timer:
    def __init__(self):
        self.starts = {}
        self.cumm = {}

    def __del__(self):
        print("Timers:")
        for key, val in self.cumm.items():
            print(key, "\t", val)

    def Start(self, str):
        self.starts[str] = time.time()

    def Pause(self, str):
        now = time.time()
        then = self.starts[str]

        if str in self.cumm:
            self.cumm[str] += now - then
        else:
            self.cumm[str] = now - then
        
######################################################################################
def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)
######################################################################################
class LearningParams:
    def __init__(self, saveDir, deleteDuplicateTransitions):
        self.gamma = 0.9 #0.99
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
        self.FEATURES_PER_ACTION = 2

        self.saveDir = saveDir
        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        self.corpus = Corpus(params, self)

        # These lines establish the feed-forward part of the network used to choose actions
        INPUT_DIM = 20
        EMBED_DIM = INPUT_DIM * params.NUM_ACTIONS * params.FEATURES_PER_ACTION
        #print("INPUT_DIM", INPUT_DIM, EMBED_DIM)
        
        HIDDEN_DIM = 128

        # EMBEDDINGS
        ns = len(env.nodes)
        self.embeddings = tf.Variable(tf.random_uniform([ns, INPUT_DIM], 0, 0.01))
        #print("self.embeddings", self.embeddings)

        self.input = tf.placeholder(shape=[None, params.NUM_ACTIONS * params.FEATURES_PER_ACTION], dtype=tf.int32)
        #print("self.input", self.input)

        self.embedding = tf.nn.embedding_lookup(self.embeddings, self.input)
        self.embedding = tf.reshape(self.embedding, [tf.shape(self.input)[0], EMBED_DIM])

        # SIBLINGS
        #self.siblings = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.int32)
        self.siblings = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.float32)

        # HIDDEN 1
        self.hidden1 = tf.concat([self.embedding, self.siblings], 1) 

        self.Whidden1 = tf.Variable(tf.random_uniform([EMBED_DIM + params.NUM_ACTIONS, EMBED_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.hidden1, self.Whidden1)

        self.BiasHidden1 = tf.Variable(tf.random_uniform([1, EMBED_DIM], 0, 0.01))
        self.hidden1 = tf.add(self.hidden1, self.BiasHidden1)

        self.hidden1 = tf.math.l2_normalize(self.hidden1, axis=1)
        #self.hidden1 = tf.nn.relu(self.hidden1)

        # HIDDEN 2
        self.hidden2 = self.hidden1

        self.Whidden2 = tf.Variable(tf.random_uniform([EMBED_DIM, HIDDEN_DIM], 0, 0.01))
        self.hidden2 = tf.matmul(self.hidden2, self.Whidden2)

        self.BiasHidden2 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.hidden2 = tf.add(self.hidden2, self.BiasHidden2)

        # OUTPUT
        self.Wout = tf.Variable(tf.random_uniform([HIDDEN_DIM, params.NUM_ACTIONS], 0, 0.01))

        self.Qout = tf.matmul(self.hidden2, self.Wout)

        self.predict = tf.argmax(self.Qout, 1)

        self.sumWeight = tf.reduce_sum(self.Wout) \
                        + tf.reduce_sum(self.BiasHidden2) \
                        + tf.reduce_sum(self.Whidden2) \
                        + tf.reduce_sum(self.Whidden1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer() #learning_rate=lrn_rate)
        
        self.updateModel = self.trainer.minimize(self.loss)

    def PrintQ(self, urlId, params, env, sess):
        #print("hh", urlId, env.nodes)
        visited = set()
        unvisited = Candidates()

        node = env.nodes[urlId]
        unvisited.AddLinks(env, node.urlId, visited, params)
        featuresNP, siblings = unvisited.GetFeaturesNP(env, params, visited)

        #action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: childIds})
        action, allQ = self.Predict(sess, featuresNP, siblings)
        
        #print("   curr=", curr, "action=", action, "allQ=", allQ, childIds)
        print(urlId, action, unvisited.urlIds, allQ, featuresNP)

    def PrintAllQ(self, params, env, sess):
        print("State         Q-values                          Next state")
        for node in env.nodes.values():
            urlId = node.urlId
            self.PrintQ(urlId, params, env, sess)

    def Predict(self, sess, input, siblings):
        #print("input",input.shape)
        action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: input, self.siblings: siblings})
        action = action[0]
        
        return action, allQ

    def Update(self, sess, input, siblings, targetQ):
        _, loss, sumWeight = sess.run([self.updateModel, self.loss, self.sumWeight], feed_dict={self.input: input, self.siblings: siblings, self.nextQ: targetQ})
        return loss, sumWeight

######################################################################################
class Qnets():
    def __init__(self, params, env):
        self.q = []
        self.q.append(Qnetwork(params, env))
        self.q.append(Qnetwork(params, env))

######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.qn = qn
        self.transitions = []
        self.losses = []
        self.sumWeights = []

    def AddTransition(self, transition, deleteDuplicateTransitions):
        if deleteDuplicateTransitions:
            for currTrans in self.transitions:
                if currTrans.currURLId == transition.currURLId and currTrans.nextURLId == transition.nextURLId:
                    return
            # completely new trans
    
        self.transitions.append(transition)

    def AddPath(self, path, deleteDuplicateTransitions):
        for transition in path:
            self.AddTransition(transition, deleteDuplicateTransitions)


    def GetBatch(self, maxBatchSize):        
        batch = self.transitions[0:maxBatchSize]
        self.transitions = self.transitions[maxBatchSize:]

        return batch

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def GetStopFeaturesNP(self, params):
        features = np.zeros([1, params.NUM_ACTIONS])
        return features

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
        features = np.empty([batchSize, params.NUM_ACTIONS * params.FEATURES_PER_ACTION], dtype=np.int)
        siblings = np.empty([batchSize, params.NUM_ACTIONS], dtype=np.int)
        targetQ = np.empty([batchSize, params.NUM_ACTIONS])

        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next

            features[i, :] = transition.features
            targetQ[i, :] = transition.targetQ
            siblings[i, :] = transition.siblings
        
            i += 1

        #_, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
        timer.Start("UpdateQN.1")
        loss, sumWeight = self.qn.Update(sess, features, siblings, targetQ)
        timer.Pause("UpdateQN.1")

        #print("loss", loss)
        return loss, sumWeight

######################################################################################
# helpers
class MySQL:
    def __init__(self):
        # paracrawl
        self.mydb = mysql.connector.connect(
        host="localhost",
        user="paracrawl_user",
        passwd="paracrawl_password",
        database="paracrawl",
        charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)

######################################################################################
def DebugTransitions(transitions):
    ret = ""
    for transition in transitions:
        str = transition.Debug()
        ret += str + " "
    return ret

class Transition:
    def __init__(self, currURLId, nextURLId, done, features, siblings, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.done = done
        self.features = features 
        self.siblings = siblings
        self.targetQ = targetQ

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

class Env:
    def __init__(self, sqlconn, url):
        self.url = url
        self.numAligned = 0
        self.nodes = {}
        self.url2urlId = {}
        self.docId2URLIds = {}

        # all nodes with docs
        sql = "select url.id, url.document_id, document.lang_id, url.val from url, document where url.document_id = document.id and val like %s"
        val = (url + "%",)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        for rec in res:
            #print("rec", rec[0], rec[1], rec[2], rec[3])
            url = rec[3]
            if url[-10:] == "robots.txt":
                #print("Skipping robots.txt")
                continue

            node = Node(sqlconn, rec[0], rec[1], rec[2], url)
            self.nodes[node.urlId] = node
            self.url2urlId[node.url] = node.urlId
            self.AddDocId(node.docId, node.urlId)

            if node.alignedDoc > 0:
                self.numAligned += 1
        print("numAligned", self.numAligned)

        for node in self.nodes.values():
            node.CreateLinks(sqlconn, self)
        
        self.CreateStartNodes(sqlconn)

        # stop node
        node = Node(sqlconn, 0, 0, 0, "STOP")
        #self.nodesbyURL[node.url] = node
        self.nodes[0] = node

        for node in self.nodes.values():
            print(node.Debug())
        print("all nodes", len(self.nodes))

        # print out
        #for node in self.nodes:
        #    print("node", node.Debug())

        #node = Node(sqlconn, url, True)
        #print("node", node.docId, node.urlId)       

    def CreateStartNodes(self, sqlconn):
        # start node id = sys.maxsize
        startNode = Node(sqlconn, sys.maxsize, 0, 0, "START")

        graphs = self.CreateGraphs()
        print("#graphs", len(graphs))
        for graph in graphs:
            print("   ", graph.urlId, graph.url)
            startNode.CreateLink("", 0, graph)

        self.nodes[startNode.urlId] = startNode
        print("startNode", startNode.Debug())

    def CreateGraphs(self):
        graphs = []

        i = 0
        for node in self.nodes.values():
            #print("node", i, node.urlId, node.url)
            found = False
            for graph in graphs:
                #print("   graph", graph.urlId, graph.url)
                visited = set()
                found = self.Search(graph, node, visited)
                if found:
                    break
            
            if not found:
                print("   graphs.append", node.urlId, node.url)
                graphs.append(node)

            i += 1
        
        return graphs

    def Search(self, graph, node, visited):
        if graph.urlId in visited:
            #print("   already visited", graph.urlId, visited)
            return False
        else:
            visited.add(graph.urlId)

        #print("   graph.urlId", graph.urlId, node.urlId, len(graph.links))
        if graph.urlId == node.urlId:
            #print("   found", node.urlId)
            return True

        for link in graph.links:
            childGraph = link.childNode
            #print("   recursive search", childGraph.urlId)
            found = self.Search(childGraph, node, visited)
            if found:
                #print("   FOUND", graph.urlId, node.urlId)
                return True

        return False

    def __del__(self):
        pass

    def GetURLIdFromURL(self, url):
        if url in self.url2urlId:
            return self.url2urlId[url]

        raise Exception("URL not found:" + url)

    def AddDocId(self, docId, urlId):
        if docId in self.docId2URLIds:
            self.docId2URLIds[docId].add(urlId)
        else:
            urlIds = set()
            urlIds.add(urlId)
            self.docId2URLIds[docId] = urlIds

    def GetURLIdsFromDocId(self, docId):
        if docId in self.docId2URLIds:
            return self.docId2URLIds[docId]

        raise Exception("Doc id not found:" + str(docId))

    def GetNextState(self, action, visited, unvisited, docsVisited):
        #nextNodeId = childIds[0, action]
        nextURLId = unvisited.GetNextState(action)
        #print("   nextNodeId", nextNodeId)
        nextNode = self.nodes[nextURLId]
        docId = nextNode.docId
        if nextURLId == 0:
            #print("   stop")
            reward = 0.0
        elif nextNode.alignedDoc > 0:
            reward = -1.0

            # has this doc been crawled?
            if docId not in docsVisited:
                # has the other doc been crawled?
                urlIds = self.GetURLIdsFromDocId(nextNode.alignedDoc)
                for urlId in urlIds:
                    if urlId in visited:
                        reward = 17.0 #170.0
                        break
            #print("   visited", visited)
            #print("   nodeIds", nodeIds)
            #print("   reward", reward)
            #print()
        else:
            #print("   non-rewarding")
            reward = -1.0

        return nextURLId, docId, reward

    def Walk(self, start, params, sess, qn, printQ):
        visited = set()
        unvisited = Candidates()
        docsVisited = set()
        
        curr = start
        i = 0
        numAligned = 0
        totReward = 0.0
        totDiscountedReward = 0.0
        discount = 1.0
        mainStr = str(curr) + "->"
        rewardStr = "  ->"
        debugStr = ""

        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            currNode = self.nodes[curr]
            unvisited.AddLinks(self, currNode.urlId, visited, params)
            featuresNP, siblings = unvisited.GetFeaturesNP(self, params, visited)
            #print("featuresNP", featuresNP)
            #print("siblings", siblings)

            if printQ: unvisitedStr = str(unvisited.urlIds)

            action, allQ = qn.Predict(sess, featuresNP, siblings)
            nextURLId, nextDocId, reward = self.GetNextState(action, visited, unvisited, docsVisited)
            totReward += reward
            totDiscountedReward += discount * reward
            visited.add(nextURLId)
            unvisited.RemoveLink(nextURLId)
            docsVisited.add(nextDocId)

            alignedStr = ""
            nextNode = self.nodes[nextURLId]
            if nextNode.alignedDoc > 0:
                alignedStr = "*"
                numAligned += 1

            if printQ:
                debugStr += "   " + str(curr) + "->" + str(nextURLId) + " " \
                         + str(action) + " " + unvisitedStr + " " \
                         + str(allQ) + " " \
                         + str(featuresNP) + " " \
                         + "\n"

            #print("(" + str(action) + ")", str(nextURLId) + "(" + str(reward) + ") -> ", end="")
            mainStr += str(nextURLId) + alignedStr + "->"
            rewardStr += str(reward) + "->"
            curr = nextURLId
            discount *= params.gamma
            i += 1

            if nextURLId == 0: break

        mainStr += " " + str(i) + "/" + str(numAligned)
        rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

        if printQ:
            print(debugStr, end="")
        print(mainStr)
        print(rewardStr)

        return numAligned, totReward, totDiscountedReward



    def WalkAll(self, params, sess, qn):
        for node in self.nodes.values():
            self.Walk(node.urlId, params, sess, qn, False)

    def GetNumberAligned(self, path):
        ret = 0
        for transition in path:
            next = transition.nextURLId
            nextNode = self.nodes[next]
            if nextNode.alignedDoc > 0:
                ret += 1
        return ret


    def Neural(self, epoch, currURLId, params, sess, qnA, qnB, visited, unvisited, docsVisited):
        timer.Start("Neural.1")
        #DEBUG = False

        unvisited.AddLinks(self, currURLId, visited, params)
        featuresNP, siblings = unvisited.GetFeaturesNP(self, params, visited)
        nextStates = unvisited.GetNextStates(params)
        #print("   childIds", childIds, unvisited)
        timer.Pause("Neural.1")

        timer.Start("Neural.2")
        action, Qs = qnA.Predict(sess, featuresNP, siblings)
        if np.random.rand(1) < params.eps:
            #if DEBUG: print("   random")
            action = np.random.randint(0, params.NUM_ACTIONS)
        timer.Pause("Neural.2")
        
        timer.Start("Neural.3")
        nextURLId, nextDocId, r = self.GetNextState(action, visited, unvisited, docsVisited)
        nextNode = self.nodes[nextURLId]
        #if DEBUG: print("   action", action, next, Qs)
        timer.Pause("Neural.3")

        timer.Start("Neural.4")
        visited.add(nextURLId)
        unvisited.RemoveLink(nextURLId)
        nextUnvisited = unvisited.copy()
        docsVisited.add(nextDocId)
        timer.Pause("Neural.4")

        timer.Start("Neural.5")
        if nextURLId == 0:
            done = True
            maxNextQ = 0.0
        else:
            assert(nextURLId != 0)
            done = False

            # Obtain the Q' values by feeding the new state through our network
            nextUnvisited.AddLinks(self, nextNode.urlId, visited, params)
            nextFeaturesNP, nextSiblings = nextUnvisited.GetFeaturesNP(self, params, visited)
            nextAction, nextQs = qnA.Predict(sess, nextFeaturesNP, nextSiblings)        
            #print("  nextAction", nextAction, nextQ)

            #assert(qnB == None)
            #maxNextQ = np.max(nextQs)

            _, nextQsB = qnB.Predict(sess, nextFeaturesNP, nextSiblings)        
            maxNextQ = nextQsB[0, nextAction]
        timer.Pause("Neural.5")
            
        timer.Start("Neural.6")
        targetQ = Qs
        #targetQ = np.array(Qs, copy=True)
        #print("  targetQ", targetQ)
        newVal = r + params.gamma * maxNextQ
        targetQ[0, action] = (1 - params.alpha) * targetQ[0, action] + params.alpha * newVal
        #targetQ[0, action] = newVal
        self.ZeroOutStop(targetQ, nextStates)

        #if DEBUG: print("   nextStates", nextStates)
        #if DEBUG: print("   targetQ", targetQ)

        transition = Transition(currURLId, nextNode.urlId, done, np.array(featuresNP, copy=True), np.array(siblings, copy=True), np.array(targetQ, copy=True))
        timer.Pause("Neural.6")

        return transition

    def Trajectory(self, epoch, currURLId, params, sess, qns):
        visited = set()
        unvisited = Candidates()
        docsVisited = set()

        while (True):
            tmp = np.random.rand(1)
            if tmp > 0.5:
                qnA = qns.q[0]
                qnB = qns.q[1]
            else:
                qnA = qns.q[1]
                qnB = qns.q[0]
            #qnA = qns.q[0]
            #qnB = None

            transition = self.Neural(epoch, currURLId, params, sess, qnA, qnB, visited, unvisited, docsVisited)
            
            qnA.corpus.AddTransition(transition, params.deleteDuplicateTransitions)

            currURLId = transition.nextURLId
            #print("visited", visited)

            if transition.done: break
        #print("unvisited", unvisited)
        
    def ZeroOutStop(self, targetQ, nextStates):
        assert(targetQ.shape == nextStates.shape)

        i = 0
        for i in range(nextStates.shape[1]):
            if nextStates[0, i] == 0:
                targetQ[0, i] = 0

######################################################################################
class Link:
    def __init__(self, text, textLang, parentNode, childNode):
        self.text = text 
        self.textLang = textLang 
        self.parentNode = parentNode
        self.childNode = childNode

class Node:
    def __init__(self, sqlconn, urlId, docId, lang, url):

        self.urlId = urlId
        self.docId = docId
        self.lang = lang
        self.url = url
        self.links = []
        self.alignedDoc = 0

        self.CreateAlign(sqlconn)

    def CreateAlign(self, sqlconn):
        if self.docId is not None:
            sql = "select document1, document2 from document_align where document1 = %s or document2 = %s"
            val = (self.docId,self.docId)
            #print("sql", sql)
            sqlconn.mycursor.execute(sql, val)
            res = sqlconn.mycursor.fetchall()
            #print("alignedDoc",  self.url, self.docId, res)

            if len(res) > 0:
                rec = res[0]
                if self.docId == rec[0]:
                    self.alignedDoc = rec[1]
                elif self.docId == rec[1]:
                    self.alignedDoc = rec[0]
                else:
                    assert(True)

        #print(self.Debug())

    def Debug(self):
        strLinks = ""
        for link in self.links:
            strLinks += str(link.childNode.urlId) + " "

        return " ".join([str(self.urlId), StrNone(self.docId), 
                        StrNone(self.lang), str(self.alignedDoc), self.url,
                        "links=", str(len(self.links)), ":", strLinks ] )

    def CreateLinks(self, sqlconn, env):
        #sql = "select id, text, url_id from link where document_id = %s"
        sql = "select link.id, link.text, link.text_lang_id, link.url_id, url.val from link, url where url.id = link.url_id and link.document_id = %s"
        val = (self.docId,)
        #print("sql", sql)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        for rec in res:
            text = rec[1]
            textLang = rec[2]
            urlId = rec[3]
            url = rec[4]
            #print("urlid", self.docId, text, urlId)

            if urlId in env.nodes:
                childNode = env.nodes[urlId]
                #print("child", self.docId, childNode.Debug())
            else:
                continue
                #childNode = Node(sqlconn, urlId, None, None, url)
                #nodes[childNode.urlId] = childNode
                #nodesbyURL[childNode.url] = childNode
                #nodes.append(childNode)

            self.CreateLink(text, textLang, childNode)

    def CreateLink(self, text, textLang, childNode):            
        link = Link(text, textLang, self, childNode)
        self.links.append(link)

    def GetLinks(self, visited, params):
        ret = []
        for link in self.links:
            childNode = link.childNode
            childURLId = childNode.urlId
            #print("   ", childNode.Debug())
            if childURLId != self.urlId and childURLId not in visited:
                ret.append(link)
        #print("   childIds", childIds)

        return ret

######################################################################################

class Candidates:
    def __init__(self):
        self.dict = {} # nodeid -> link
        self.urlIds = []

        self.dict[0] = []
        self.urlIds.append(0)

    def AddLink(self, link):
        urlLId = link.childNode.urlId
        if urlLId not in self.dict:
            self.dict[urlLId] = []
            self.urlIds.append(link.childNode.urlId)
        self.dict[urlLId].append(link)

    def RemoveLink(self, nextURLId):
        del self.dict[nextURLId]
        self.urlIds.remove(nextURLId)

    def copy(self):
        ret = Candidates()
        ret.dict = self.dict.copy()
        ret.urlIds = self.urlIds.copy()

        return ret

    def AddLinks(self, env, urlId, visited, params):
        currNode = env.nodes[urlId]
        #print("   currNode", curr, currNode.Debug())
        newLinks = currNode.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

    def GetFeaturesNP(self, env, params, visited):
        ret = np.zeros([params.NUM_ACTIONS, params.FEATURES_PER_ACTION], dtype=np.int)
        siblings = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for urlId in self.urlIds:
            #ret[0, i] = childId

            links = self.dict[urlId]
            if len(links) > 0:
                link = links[0]
                #print("link", link.parentNode.urlId, link.childNode.urlId, link.text, link.textLang)
                ret[i, 0] = link.textLang

                parentNode = link.parentNode
                #print("parentNode", childId, parentNode.lang, parentLangId, parentNode.Debug())
                ret[i, 1] = parentNode.lang

                matchedSiblings = self.GetMatchedSiblings(env, urlId, parentNode, visited)
                numMatchedSiblings = len(matchedSiblings)
                #if numMatchedSiblings > 1:
                #    print("matchedSiblings", urlId, parentNode.urlId, matchedSiblings, visited)
                
                siblings[0, i] = numMatchedSiblings
                
            i += 1
            if i >= params.NUM_ACTIONS:
                #print("overloaded", len(self.dict), self.dict)
                break

        #print("BEFORE", ret)
        ret = ret.reshape([1, params.NUM_ACTIONS * params.FEATURES_PER_ACTION])
        #print("AFTER", ret)
        #print()

        return ret, siblings

    def GetMatchedSiblings(self, env, urlId, parentNode, visited):
        ret = []
        numSiblings = 0

        #print("parentNode", urlId)
        for link in parentNode.links:
            sibling = link.childNode
            if sibling.urlId != urlId:
                #print("   link", sibling.urlId, sibling.alignedDoc)
                numSiblings += 1

                if sibling.urlId in visited:
                    # has the sibling AND it's matched doc been crawled?
                    if sibling.alignedDoc > 0:
                        matchedURLIds = env.GetURLIdsFromDocId(sibling.alignedDoc)
                        intersect = matchedURLIds & visited
                        if len(intersect) > 0:
                            ret.append(sibling.urlId)      

        return ret

    def GetNextState(self, action):
        if action >= len(self.urlIds):
            ret = 0
        else:
            #print("action", action, len(self.urlIds), self.urlIds)
            ret = self.urlIds[action]
        return ret

    def GetNextStates(self, params):
        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for urlId in self.urlIds:
            ret[0, i] = urlId
            i += 1
            if i >= params.NUM_ACTIONS:
                break


        return ret

######################################################################################

def Train(params, sess, saver, env, qns):
    totRewards = []
    totDiscountedRewards = []

    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        #startState = 30
        
        timer.Start("Trajectory")
        env.Trajectory(epoch, sys.maxsize, params, sess, qns)
        timer.Pause("Trajectory")

        timer.Start("Update")
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)
        timer.Pause("Update")

        if epoch > 0 and epoch % params.walk == 0:
            if len(qns.q[0].corpus.losses) > 0:
                # trained at least once
                #qns.q[0].PrintAllQ(params, env, sess)
                qns.q[0].PrintQ(0, params, env, sess)
                qns.q[0].PrintQ(sys.maxsize, params, env, sess)
                print()

                numAligned, totReward, totDiscountedReward = env.Walk(sys.maxsize, params, sess, qns.q[0], True)
                totRewards.append(totReward)
                totDiscountedRewards.append(totDiscountedReward)
                print("epoch", epoch, "loss", qns.q[0].corpus.losses[-1], "eps", params.eps, "alpha", params.alpha)
                print()
                sys.stdout.flush()

                #saver.save(sess, "{}/hh".format(params.saveDir), global_step=epoch)

                #numAligned = env.GetNumberAligned(path)
                #print("path", numAligned, env.numAligned)
                if numAligned >= env.numAligned - 5:
                    #print("got them all!")
                    #eps = 1. / ((i/50) + 10)
                    params.eps *= .99
                    params.eps = max(0.1, params.eps)
                    
                    #params.alpha *= 0.99
                    #params.alpha = max(0.3, params.alpha)
                
            #print("epoch", epoch, \
            #     len(qns.q[0].corpus.transitions), len(qns.q[1].corpus.transitions)) #, \
            #     #DebugTransitions(qns.q[0].corpus.transitions))
                

    return totRewards, totDiscountedRewards
            
######################################################################################
 
timer = Timer()
 
def Main():
    print("Starting")

    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--save-dir", dest="saveDir", default=".",
                     help="Directory that model WIP are saved to. If existing model exists then load it")
    oparser.add_argument("--delete-duplicate-transitions", dest="deleteDuplicateTransitions", default=False,
                     help="If True then only unique transition are used in each batch")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    global timer

    sqlconn = MySQL()

    #hostName = "vade-retro.fr"
    #hostName = "www.visitbritain.com"
    hostName = "www.buchmann.ch"
    pickleName = hostName + ".pickle"

    if os.path.exists(pickleName):
        with open(pickleName, 'rb') as f:
            print("unpickling")
            env = pickle.load(f)
    else:
        env = Env(sqlconn, hostName)
        with open(pickleName, 'wb') as f:
            print("pickling")
            pickle.dump(env,f)
        

    params = LearningParams(options.saveDir, options.deleteDuplicateTransitions)

    tf.reset_default_graph()
    qns = Qnets(params, env)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)

        qns.q[0].PrintAllQ(params, env, sess)
        #env.WalkAll(params, sess, qn)
        print()

        timer.Start("Train")
        totRewards, totDiscountedRewards = Train(params, sess, saver, env, qns)
        timer.Pause("Train")
        
        #qn.PrintAllQ(params, env, sess)
        #env.WalkAll(params, sess, qn)

        env.Walk(sys.maxsize, params, sess, qns.q[0], True)

        del timer

        plt.plot(totRewards)
        plt.plot(totDiscountedRewards)
        plt.show()

        plt.plot(qns.q[0].corpus.losses)
        plt.plot(qns.q[1].corpus.losses)
        plt.show()

        plt.plot(qns.q[0].corpus.sumWeights)
        plt.plot(qns.q[1].corpus.sumWeights)
        plt.show()

    print("Finished")

if __name__ == "__main__":
    Main()
