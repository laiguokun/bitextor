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
import hashlib

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

def NormalizeURL(url):
    url = url.lower()
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    #if url[-5:] == ".html":
    #    url = url[:-5] + ".htm"
    #if url[-9:] == "index.htm":
    #    url = url[:-9]
    if url[-1:] == "/":
        url = url[:-1]

    if url[:7] == "http://":
        #print("   strip protocol1", url, url[7:])
        url = url[7:]
    elif url[:8] == "https://":
        #print("   strip protocol2", url, url[8:])
        url = url[8:]

    return url
######################################################################################
class LearningParams:
    def __init__(self, saveDir, deleteDuplicateTransitions):
        self.gamma = 1.0 #0.99
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 50001
        self.eps = 1 # 0.7
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 10
        
        self.debug = False
        self.walk = 1000
        self.NUM_ACTIONS = 100
        self.FEATURES_PER_ACTION = 2

        self.saveDir = saveDir
        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 1000.0 #17.0
        self.cost = -1.0
        
######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        self.corpus = Corpus(params, self)

        # These lines establish the feed-forward part of the network used to choose actions
        INPUT_DIM = 20
        EMBED_DIM = INPUT_DIM * params.NUM_ACTIONS * params.FEATURES_PER_ACTION
        #print("INPUT_DIM", INPUT_DIM, EMBED_DIM)
        
        HIDDEN_DIM = 1024

        # EMBEDDINGS
        self.embeddings = tf.Variable(tf.random_uniform([env.maxLangId + 1, INPUT_DIM], 0, 0.01))
        #print("self.embeddings", self.embeddings)

        self.input = tf.placeholder(shape=[None, params.NUM_ACTIONS * params.FEATURES_PER_ACTION], dtype=tf.int32)
        #print("self.input", self.input)

        self.embedding = tf.nn.embedding_lookup(self.embeddings, self.input)
        self.embedding = tf.reshape(self.embedding, [tf.shape(self.input)[0], EMBED_DIM])

        # NUMBER OF SIBLINGS
        self.siblings = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.float32)

        # NUMBER OF NODES
        self.numNodes = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # number of possible action. <= NUM_ACTIONS
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # HIDDEN 1
        self.hidden1 = tf.concat([self.embedding, self.siblings, self.numNodes, self.numActions], 1) 

        self.Whidden1 = tf.Variable(tf.random_uniform([EMBED_DIM + params.NUM_ACTIONS + 2, EMBED_DIM], 0, 0.01))
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

        urlIds, numURLs, featuresNP, siblings, numNodes = unvisited.GetFeaturesNP(env, params, visited)
        #print("featuresNP", featuresNP)
        
        #action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: childIds})
        action, allQ = self.Predict(sess, featuresNP, siblings, numNodes, numURLs)
        
        #print("   curr=", curr, "action=", action, "allQ=", allQ, childIds)
        numURLsScalar = int(numURLs[0,0])
        urlIdsTruncate = urlIds[0, 0:numURLsScalar]                

        print(urlId, node.url, action, numURLsScalar, urlIdsTruncate, allQ, featuresNP)

    def PrintAllQ(self, params, env, sess):
        print("State URL action unvisited  Q-values features")
        for node in env.nodes.values():
            urlId = node.urlId
            self.PrintQ(urlId, params, env, sess)

    def Predict(self, sess, input, siblings, numNodes, numURLs):
        #print("input",input.shape, siblings.shape, numNodes.shape)
        #print("numURLs", numURLs)
        action, allQ = sess.run([self.predict, self.Qout], 
                                feed_dict={self.input: input, 
                                        self.siblings: siblings, 
                                        self.numNodes: numNodes,
                                        self.numActions: numURLs})
        action = action[0]
        
        return action, allQ

    def Update(self, sess, input, siblings, numNodes, numURLs, targetQ):
        #print("numURLs", numURLs.shape)
        _, loss, sumWeight = sess.run([self.updateModel, self.loss, self.sumWeight], 
                                    feed_dict={self.input: input, 
                                            self.siblings: siblings, 
                                            self.numNodes: numNodes, 
                                            self.numActions: numURLs,
                                            self.nextQ: targetQ})
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
        numNodes = np.empty([batchSize, 1])
        numURLs = np.empty([batchSize, 1])

        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next

            features[i, :] = transition.features
            targetQ[i, :] = transition.targetQ
            siblings[i, :] = transition.siblings
            numNodes[i, :] = transition.numNodes
            numURLs[i, :] = transition.numURLs

            i += 1

        #_, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
        timer.Start("UpdateQN.1")
        loss, sumWeight = self.qn.Update(sess, features, siblings, numNodes, numURLs, targetQ)
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

class Transition:
    def __init__(self, currURLId, nextURLId, done, features, siblings, numNodes, numURLs, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.done = done
        self.features = features 
        self.siblings = siblings
        self.numNodes = numNodes
        self.numURLs = numURLs
        self.targetQ = targetQ

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

######################################################################################
class Link:
    def __init__(self, text, textLang, parentNode, childNode):
        self.text = text 
        self.textLang = textLang 
        self.parentNode = parentNode
        self.childNode = childNode
######################################################################################

class Node:
    def __init__(self, urlId, url, docIds, langIds, redirectId):
        assert(len(docIds) == len(langIds))
        self.urlId = urlId
        self.url = url
        self.docIds = set(docIds)

        self.redirectId = redirectId
        self.redirect = None

        self.links = set()
        self.lang = 0 if len(langIds) == 0 else langIds[0]
        self.alignedURLId = 0

        self.normURL = None

        #print("self.lang", self.lang, langIds, urlId, url, docIds)
        #for lang in langIds:
        #    assert(self.lang == lang)

    def CreateLink(self, text, textLang, childNode):            
        link = Link(text, textLang, self, childNode)
        self.links.add(link)

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

    def Recombine(self, loserNode):
        assert (loserNode is not None)
        # print("Recombining")
        # print("   ", self.Debug())
        # print("   ", loserNode.Debug())

        self.docIds.update(loserNode.docIds)
        self.links.update(loserNode.links)

        if self.lang == 0:
            if loserNode.lang != 0:
                self.lang = loserNode.lang
        else:
            if loserNode.lang != 0:
                assert (self.lang == loserNode.lang)

        if self.alignedURLId == 0:
            if loserNode.alignedURLId != 0:
                self.alignedURLId = loserNode.alignedURLId
        else:
            if loserNode.alignedURLId != 0:
                assert (self.alignedURLId == loserNode.alignedURLId)

    def Debug(self):
        return " ".join([str(self.urlId), self.url, StrNone(self.docIds),
                        StrNone(self.lang), StrNone(self.alignedURLId),
                        StrNone(self.redirect), str(len(self.links)),
                        StrNone(self.normURL) ] )

######################################################################################
class Env:
    def __init__(self, sqlconn, url):
        self.url = url
        self.numAligned = 0
        self.nodes = {} # urlId -> Node
        self.url2urlId = {}
        self.docId2URLIds = {}
        self.maxLangId = 0

        unvisited = {} # urlId -> Node
        visited = {} # urlId -> Node
        rootURLId = self.Url2UrlId(sqlconn, url)
        rootNode = self.CreateNode(sqlconn, visited, unvisited, rootURLId, url)
        self.CreateGraphFromDB(sqlconn, visited, unvisited)
        print("CreateGraphFromDB", len(visited))
        #for node in visited.values():
        #    print(node.Debug())

        self.ImportURLAlign(sqlconn, visited)

        #print("rootNode", rootNode.Debug())
        print("Recombine")
        normURL2Node = {}
        self.Recombine(visited, normURL2Node)
        
        rootNode = normURL2Node[rootNode.normURL]
        assert(rootNode is not None)
        print("rootNode", rootNode.Debug())

        visited = set() # set of nodes
        self.PruneEmptyNodes(rootNode, visited)

        startNode = Node(sys.maxsize, "START", [], [], None)
        startNode.CreateLink("", 0, rootNode)
        self.nodes[startNode.urlId] = startNode

        # stop node
        node = Node(0, "STOP", [], [], None)
        self.nodes[0] = node

        self.Visit(rootNode)
        print("self.nodes", len(self.nodes), self.numAligned)
        #for node in self.nodes.values():
        #    print(node.Debug())

        print("graph created")

    def GetRedirectedNormURL(self, node):
        while node.redirect is not None:
            node =  node.redirect
        normURL = NormalizeURL(node.url)
        return normURL

    def Recombine(self, visited, normURL2Node):
        #print("visited", visited.keys())
        # create winning node for each norm url
        for node in visited.values():
            node.normURL = self.GetRedirectedNormURL(node)
            if node.normURL not in normURL2Node:
                normURL2Node[node.normURL] = node
            else:
                winner = normURL2Node[node.normURL]
                winner.Recombine(node)

        # relink child nodes to winning nodes
        for node in visited.values():
            for link in node.links:
                childNode = link.childNode
                #print("childNode", childNode.Debug())
                newChildNode = normURL2Node[childNode.normURL]
                #print("newChildNode", newChildNode.Debug())
                #print()
                link.childNode = newChildNode

    def CreateNode(self, sqlconn, visited, unvisited, urlId, url):
        if urlId in visited:
            return visited[urlId]
        elif urlId in unvisited:
            return unvisited[urlId]
        else:
            docIds, langIds, redirectId = self.UrlId2Responses(sqlconn, urlId)
            node = Node(urlId, url, docIds, langIds, redirectId)
            assert(urlId not in visited)
            assert(urlId not in unvisited)
            unvisited[urlId] = node
            return node

    def CreateGraphFromDB(self, sqlconn, visited, unvisited):
        while len(unvisited) > 0:
            (urlId, node) = unvisited.popitem()
            visited[node.urlId] = node
            #print("node", node.Debug())
            assert(node.urlId == urlId)

            if node.redirectId is not None:
                assert(len(node.docIds) == 0)
                redirectURL = self.UrlId2Url(sqlconn, node.redirectId)
                redirectNode = self.CreateNode(sqlconn, visited, unvisited, node.redirectId, redirectURL)
                node.redirect = redirectNode
            else:
                linksStruct = self.DocIds2Links(sqlconn, node.docIds)

                for linkStruct in linksStruct:
                    childURLId = linkStruct[0]
                    childUrl = self.UrlId2Url(sqlconn, childURLId)
                    childNode = self.CreateNode(sqlconn, visited, unvisited, childURLId, childUrl)
                    link = Link(linkStruct[1], linkStruct[2], node, childNode)
                    node.links.add(link)

            #print("   ", node.Debug())

    def ImportURLAlign(self, sqlconn, visited):
        #print("visited", visited.keys())
        sql = "SELECT id, url1, url2 FROM url_align"
        val = ()
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            urlId1 = res[1]
            urlId2 = res[2]
            #print("urlId", urlId1, urlId2)

            _, _, redirectId = self.UrlId2Responses(sqlconn, urlId1)
            if redirectId is not None:
                urlId1 = redirectId

            _, _, redirectId = self.UrlId2Responses(sqlconn, urlId2)
            if redirectId is not None:
                urlId2 = redirectId

            #print("   ", urlId1, urlId2)
            if urlId1 not in visited or urlId2 not in visited:
                print("Alignment not in graph", urlId1, urlId2)
                continue

            node1 = visited[urlId1]
            node2 = visited[urlId2]
            node1.alignedURLId = urlId2
            node2.alignedURLId = urlId1

    def Visit(self, node):
        if node.urlId in self.nodes:
            return
        self.nodes[node.urlId] = node

        if node.alignedURLId > 0:
            self.numAligned += 1

        if node.lang > self.maxLangId:
            self.maxLangId = node.lang

        for link in node.links:
            if link.textLang > self.maxLangId:
                self.maxLangId = link.textLang

            childNode = link.childNode
            self.Visit(childNode)
            
    def PruneEmptyNodes(self, node, visited):
        if node in visited:
            return
        visited.add(node)

        linksCopy = set(node.links)
        for link in linksCopy:
            childNode = link.childNode
            if len(childNode.docIds) == 0:
                #print("empty", childNode.Debug())
                node.links.remove(link)

            self.PruneEmptyNodes(childNode, visited)

    def DocIds2Links(self, sqlconn, docIds):
        docIdsStr = ""
        for docId in docIds:
            docIdsStr += str(docId) + ","

        sql = "SELECT id, url_id, text, text_lang_id FROM link WHERE document_id IN (%s)"
        val = (docIdsStr,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        linksStruct = []
        for res in ress:
            struct = (res[1], res[2], res[3])
            linksStruct.append(struct)

        return linksStruct

    def UrlId2Responses(self, sqlconn, urlId):
        sql = "SELECT id, status_code, to_url_id, lang_id FROM response WHERE url_id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        docIds = []
        langIds = []
        redirectId = None
        for res in ress:
            if res[1] == 200:
                assert(redirectId == None)
                docIds.append(res[0])
                langIds.append(res[3])
            elif res[1] in (301, 302):
                assert(len(docIds) == 0)
                redirectId = res[2]

        return docIds, langIds, redirectId

    def RespId2URL(self, sqlconn, respId):
        sql = "SELECT T1.id, T1.val FROM url T1, response T2 " \
            + "WHERE T1.id = T2.url_id AND T2.id = %s"
        val = (respId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0], res[1]


    def UrlId2Url(self, sqlconn, urlId):
        sql = "SELECT val FROM url WHERE id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    def Url2UrlId(self, sqlconn, url):
        c = hashlib.md5()
        c.update(url.lower().encode())
        hashURL = c.hexdigest()

        if hashURL in self.url2urlId:
            return self.url2urlId[hashURL]

        sql = "SELECT id FROM url WHERE md5 = %s"
        val = (hashURL,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    ########################################################################
    def GetNextState(self, params, action, visited, urlIds):
        assert(urlIds.shape[1] > action)
        nextURLId = urlIds[0, action]
        #print("   nextNodeId", nextNodeId)
        nextNode = self.nodes[nextURLId]
        if nextURLId == 0:
            #print("   stop")
            reward = 0.0
        elif nextNode.alignedURLId > 0 and nextNode.alignedURLId in visited:
                reward = params.reward
            #print("   visited", visited)
            #print("   nodeIds", nodeIds)
            #print("   reward", reward)
            #print()
        else:
            #print("   non-rewarding")
            reward = params.cost

        return nextURLId, reward

    def Walk(self, start, params, sess, qn, printQ):
        visited = set()
        unvisited = Candidates()
        
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
            urlIds, numURLs, featuresNP, siblings, numNodes = unvisited.GetFeaturesNP(self, params, visited)
            #print("featuresNP", featuresNP)
            #print("siblings", siblings)

            if printQ: 
                numURLsScalar = int(numURLs[0,0])
                urlIdsTruncate = urlIds[0, 0:numURLsScalar]                
                unvisitedStr =  str(urlIdsTruncate)

            action, allQ = qn.Predict(sess, featuresNP, siblings, numNodes, numURLs)
            nextURLId, reward = self.GetNextState(params, action, visited, urlIds)
            totReward += reward
            totDiscountedReward += discount * reward
            visited.add(nextURLId)
            unvisited.RemoveLink(nextURLId)

            alignedStr = ""
            nextNode = self.nodes[nextURLId]
            if nextNode.alignedURLId > 0:
                alignedStr = "*"
                numAligned += 1

            if printQ:
                debugStr += "   " + str(curr) + "->" + str(nextURLId) + " " \
                         + str(action) + " " \
                         + str(numURLsScalar) + " " \
                         + unvisitedStr + " " \
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
        urlIds, numURLs, featuresNP, siblings, numNodes = unvisited.GetFeaturesNP(self, params, visited)
        #print("   childIds", childIds, unvisited)
        timer.Pause("Neural.1")

        timer.Start("Neural.2")
        action, Qs = qnA.Predict(sess, featuresNP, siblings, numNodes, numURLs)
        if np.random.rand(1) < params.eps:
            #if DEBUG: print("   random")
            action = np.random.randint(0, params.NUM_ACTIONS)
        timer.Pause("Neural.2")
        
        timer.Start("Neural.3")
        nextURLId, r = self.GetNextState(params, action, visited, urlIds)
        nextNode = self.nodes[nextURLId]
        #if DEBUG: print("   action", action, next, Qs)
        timer.Pause("Neural.3")

        timer.Start("Neural.4")
        visited.add(nextURLId)
        unvisited.RemoveLink(nextURLId)
        nextUnvisited = unvisited.copy()
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
            _, nextNumURLs, nextFeaturesNP, nextSiblings, nextNumNodes = nextUnvisited.GetFeaturesNP(self, params, visited)
            nextAction, nextQs = qnA.Predict(sess, nextFeaturesNP, nextSiblings, nextNumNodes, nextNumURLs)        
            #print("nextNumNodes", numNodes, nextNumNodes)
            #print("  nextAction", nextAction, nextQ)

            #assert(qnB == None)
            #maxNextQ = np.max(nextQs)

            _, nextQsB = qnB.Predict(sess, nextFeaturesNP, nextSiblings, nextNumNodes, nextNumURLs)
            maxNextQ = nextQsB[0, nextAction]
        timer.Pause("Neural.5")
            
        timer.Start("Neural.6")
        targetQ = Qs
        #targetQ = np.array(Qs, copy=True)
        #print("  targetQ", targetQ)
        newVal = r + params.gamma * maxNextQ
        targetQ[0, action] = (1 - params.alpha) * targetQ[0, action] + params.alpha * newVal
        #targetQ[0, action] = newVal
        self.ZeroOutStop(targetQ, urlIds, numURLs)

        #if DEBUG: print("   nextStates", nextStates)
        #if DEBUG: print("   targetQ", targetQ)

        transition = Transition(currURLId, 
                                nextNode.urlId, 
                                done, 
                                np.array(featuresNP, copy=True), 
                                np.array(siblings, copy=True), 
                                numNodes,
                                numURLs,
                                np.array(targetQ, copy=True))
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
        
    def ZeroOutStop(self, targetQ, urlIds, numURLs):
        #print("urlIds", numURLs, targetQ, urlIds)
        assert(targetQ.shape == urlIds.shape)
        targetQ[0,0] = 0.0
        
        #i = 0
        #for i in range(urlIds.shape[1]):
        #    if urlIds[0, i] == 0:
        #        targetQ[0, i] = 0

        numURLsScalar = int(numURLs[0,0])
        for i in range(numURLsScalar, targetQ.shape[1]):
            targetQ[0, i] = -555.0

        #print("targetQ", targetQ)
        

######################################################################################

class Candidates:
    def __init__(self):
        self.dict = {} # nodeid -> link
        self._urlIds = []

        self.dict[0] = []
        self._urlIds.append(0)

    def AddLink(self, link):
        urlLId = link.childNode.urlId
        if urlLId not in self.dict:
            self.dict[urlLId] = []
            self._urlIds.append(link.childNode.urlId)
        self.dict[urlLId].append(link)

    def RemoveLink(self, nextURLId):
        del self.dict[nextURLId]
        self._urlIds.remove(nextURLId)

    def copy(self):
        ret = Candidates()
        ret.dict = self.dict.copy()
        ret._urlIds = self._urlIds.copy()

        return ret

    def AddLinks(self, env, urlId, visited, params):
        currNode = env.nodes[urlId]
        #print("   currNode", curr, currNode.Debug())
        newLinks = currNode.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

    def _GetNextURLIds(self, params):
        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        urlIds = self._urlIds[1:]
        #random.shuffle(urlIds)
        i = 1
        for urlId in urlIds:
            ret[0, i] = urlId
            i += 1
            if i >= params.NUM_ACTIONS:
                break

        return ret, i

    def GetFeaturesNP(self, env, params, visited):
        urlIds, numURLs = self._GetNextURLIds(params)
        #print("urlIds", urlIds)

        langFeatures = np.zeros([params.NUM_ACTIONS, params.FEATURES_PER_ACTION], dtype=np.int)
        siblings = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for urlId in urlIds[0]:
            #print("urlId", urlId)
            
            links = self.dict[urlId]
            if len(links) > 0:
                link = links[0]
                #print("link", link.parentNode.urlId, link.childNode.urlId, link.text, link.textLang)
                langFeatures[i, 0] = link.textLang

                parentNode = link.parentNode
                #print("parentNode", childId, parentNode.lang, parentLangId, parentNode.Debug())
                langFeatures[i, 1] = parentNode.lang

                matchedSiblings = self.GetMatchedSiblings(env, urlId, parentNode, visited)
                numMatchedSiblings = len(matchedSiblings)
                #if numMatchedSiblings > 1:
                #    print("matchedSiblings", urlId, parentNode.urlId, matchedSiblings, visited)
                
                siblings[0, i] = numMatchedSiblings
                
            i += 1
            if i >= numURLs:
                #print("overloaded", len(self.dict), self.dict)
                break

        #print("BEFORE", ret)
        langFeatures = langFeatures.reshape([1, params.NUM_ACTIONS * params.FEATURES_PER_ACTION])
        #print("AFTER", ret)
        #print()
        numNodes = np.empty([1,1])
        numNodes[0,0] = len(visited)
        
        numURLsRet = np.empty([1,1])
        numURLsRet[0,0] = numURLs

        return urlIds, numURLsRet, langFeatures, siblings, numNodes

    def GetMatchedSiblings(self, env, urlId, parentNode, visited):
        ret = []

        #print("parentNode", urlId)
        for link in parentNode.links:
            sibling = link.childNode
            if sibling.urlId != urlId:
                #print("   link", sibling.urlId, sibling.alignedDoc)
                if sibling.urlId in visited:
                    # sibling has been crawled
                    if sibling.alignedURLId > 0 and sibling.alignedURLId in visited:
                        # sibling has been matched
                        ret.append(sibling.urlId)      

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
            
def DebugTransitions(transitions):
    ret = ""
    for transition in transitions:
        str = transition.Debug()
        ret += str + " "
    return ret

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

    #hostName = "http://vade-retro.fr/"
    hostName = "http://www.visitbritain.com/"
    #hostName = "http://www.buchmann.ch/"
    pickleName = hostName + ".pickle"

    env = Env(sqlconn, hostName)
    # if os.path.exists(pickleName):
    #     with open(pickleName, 'rb') as f:
    #         print("unpickling")
    #         env = pickle.load(f)
    # else:
    #     env = Env(sqlconn, hostName)
    #     with open(pickleName, 'wb') as f:
    #         print("pickling")
    #         pickle.dump(env,f)
        

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
