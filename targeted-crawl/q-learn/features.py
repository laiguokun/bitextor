#!/usr/bin/env python3

import numpy as np
import pylab as plt
import tensorflow as tf
import random
from collections import namedtuple
import mysql.connector
import time

######################################################################################
def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)
######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 0.9 #0.99
        self.lrn_rate = 0.1
        self.alpha = 1.0 # 0.7
        self.max_epochs = 20001
        self.eps = 0.7
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 10
        
        self.debug = False
        self.walk = 1000
        self.NUM_ACTIONS = 30

######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        self.corpus = Corpus(params, self)

        # These lines establish the feed-forward part of the network used to choose actions
        EMBED_DIM = 3000

        INPUT_DIM = EMBED_DIM // params.NUM_ACTIONS

        HIDDEN_DIM = 128

        # EMBEDDINGS
        self.embeddings = tf.Variable(tf.random_uniform([env.ns, INPUT_DIM], 0, 0.01))

        self.input = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.int32)

        self.embedding = tf.nn.embedding_lookup(self.embeddings, self.input)
        self.embedding = tf.reshape(self.embedding, [tf.shape(self.input)[0], EMBED_DIM])

        # HIDDEN 1
        self.hidden1 = self.embedding

        self.Whidden1 = tf.Variable(tf.random_uniform([EMBED_DIM, EMBED_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.hidden1, self.Whidden1)

        #self.BiasHidden1 = tf.Variable(tf.random_uniform([1, EMBED_DIM], 0, 0.01))
        #self.hidden1 = tf.add(self.hidden1, self.BiasHidden1)

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

    def PrintQ(self, curr, params, env, sess):
        # print("hh", next, hh)
        visited = set()
        unvisited = Candidates()

        unvisited.AddLinks(env, curr, visited, params)
        featuresNP = unvisited.GetFeaturesNP(env, params)

        #action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: childIds})
        action, allQ = self.Predict(sess, featuresNP)
        
        #print("   curr=", curr, "action=", action, "allQ=", allQ, childIds)
        print(curr, action, allQ, featuresNP)

    def PrintAllQ(self, params, env, sess):
        print("State         Q-values                          Next state")
        for curr in range(env.ns):
            self.PrintQ(curr, params, env, sess)

    def Predict(self, sess, input):
        action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: input})
        action = action[0]
        
        return action, allQ

    def Update(self, sess, input, targetQ):
        _, loss, sumWeight = sess.run([self.updateModel, self.loss, self.sumWeight], feed_dict={self.input: input, self.nextQ: targetQ})
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

    def AddTransition(self, transition):
        self.transitions.append(transition)

    def AddPath(self, path):
        for transition in path:
            self.AddTransition(transition)


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

    def AddStopTransition(self, env, params):
        # stop state
        for i in range(10):
            targetQ = np.zeros([1, params.NUM_ACTIONS])
            childIds = self.GetStopFeaturesNP(params)
            transition = env.Transition(0, 0, True, np.array(childIds, copy=True), np.array(targetQ, copy=True))
            self.transitions.append(transition)

    def Train(self, sess, env, params):
        if len(self.transitions) >= params.minCorpusSize:
            #self.AddStopTransition(env, params)

            for i in range(params.trainNumIter):
                batch = self.GetBatchWithoutDelete(params.maxBatchSize)
                loss, sumWeight = self.UpdateQN(params, env, sess, batch)
                self.losses.append(loss)
                self.sumWeights.append(sumWeight)
            self.transitions.clear()

    def UpdateQN(self, params, env, sess, batch):
        batchSize = len(batch)
        #print("batchSize", batchSize)
        features = np.empty([batchSize, params.NUM_ACTIONS], dtype=np.int)
        targetQ = np.empty([batchSize, params.NUM_ACTIONS])

        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next

            features[i, :] = transition.features
            targetQ[i, :] = transition.targetQ
        
            i += 1

        #_, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
        loss, sumWeight = self.qn.Update(sess, features, targetQ)

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
class Env:
    def __init__(self, sqlconn, url):
        self.Transition = namedtuple("Transition", "curr next done features targetQ")
        self.langIds = {}
        self.numAligned = 0
        self.nodes = []
        self.url2urlId = {}
        self.urlId2nodeId = {}
        self.docId2nodeIds = {}

        # stop node = 1st node in the vec
        node = Node(sqlconn, 0, 0, 0, None, "STOP")
        #self.nodesbyURL[node.url] = node
        self.nodes.append(node)

        # all nodes with docs
        sql = "select url.id, url.document_id, document.lang, url.val from url, document where url.document_id = document.id and val like %s"
        val = (url + "%",)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        for rec in res:
            #print("rec", rec[0], rec[1])
            id = len(self.nodes)
            node = Node(sqlconn, id, rec[0], rec[1], rec[2], rec[3])
            self.nodes.append(node)
            self.url2urlId[node.url] = node.urlId
            self.urlId2nodeId[node.urlId] = id
            self.AddDocId(node.docId, id)

            if node.aligned > 0:
                self.numAligned += 1
        print("numAligned", self.numAligned)

        # start node = last node in the vec
        id = len(self.nodes)
        startNode = Node(sqlconn, id, 0, 0, None, "START")

        # start node has 1 child
        nodeId = self.GetNodeIdFromURL(url)
        rootNode = self.nodes[nodeId]
        assert(rootNode is not None)
        startNode.CreateLink("", None, rootNode)

        self.nodes.append(startNode)
        self.startNodeId = startNode.id
        #print("startNode", startNode.Debug())

        self.ns = len(self.nodes) # number of states

        for node in self.nodes:
            node.CreateLinks(sqlconn, self)
            print(node.Debug())
        

        print("all nodes", len(self.nodes))

        # print out
        #for node in self.nodes:
        #    print("node", node.Debug())

        #node = Node(sqlconn, url, True)
        #print("node", node.docId, node.urlId)       

    def __del__(self):
        print("langIds", self.langIds)

    def GetLangId(self, lang):
        if lang in self.langIds:
            ret = self.langIds[lang]
        else:
            ret = len(self.langIds) + 1
            self.langIds[lang] = ret
        
        return ret

    def GetURLIdFromURL(self, url):
        if url in self.url2urlId:
            return self.url2urlId[url]

        raise Exception("URL not found:" + url)

    def GetNodeIdFromURLId(self, urlId):
        if urlId in self.urlId2nodeId:
            return self.urlId2nodeId[urlId]

        raise Exception("URL id not found:" + urlId)

    def GetNodeIdFromURL(self, url):
        urlId = self.GetURLIdFromURL(url)
        nodeId = self.GetNodeIdFromURLId(urlId)
        return nodeId

    def AddDocId(self, docId, nodeId):
        if docId in self.docId2nodeIds:
            self.docId2nodeIds[docId].add(nodeId)
        else:
            nodeIds = set()
            nodeIds.add(nodeId)
            self.docId2nodeIds[docId] = nodeIds

    def GetNodeIdsFromDocId(self, docId):
        if docId in self.docId2nodeIds:
            return self.docId2nodeIds[docId]

        raise Exception("Doc id not found:" + docId)

    def GetNextState(self, action, visited, unvisited, docsVisited):
        #nextNodeId = childIds[0, action]
        nextNodeId = unvisited.GetNextState(action)
        #print("   nextNodeId", nextNodeId)
        nextNode = self.nodes[nextNodeId]
        docId = nextNode.docId
        if nextNodeId == 0:
            #print("   stop")
            reward = 0.0
        elif nextNode.aligned > 0:
            reward = -1.0

            # has this doc been crawled?
            if docId not in docsVisited:
                # has the other doc been crawled?
                nodeIds = self.GetNodeIdsFromDocId(nextNode.aligned)
                for nodeId in nodeIds:
                    if nodeId in visited:
                        reward = 17.0
                        break
            #print("   visited", visited)
            #print("   nodeIds", nodeIds)
            #print("   reward", reward)
            #print()
        else:
            #print("   non-rewarding")
            reward = -1.0       

        return nextNodeId, docId, reward

    def Walk(self, start, params, sess, qn, printQ):
        numAligned = 0

        visited = set()
        unvisited = Candidates()
        docsVisited = set()
        
        curr = start
        i = 0
        totReward = 0
        mainStr = str(curr) + "->"
        debugStr = ""

        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            unvisited.AddLinks(self, curr, visited, params)
            featuresNP = unvisited.GetFeaturesNP(self, params)

            if printQ: unvisitedStr = str(unvisited.vec)

            action, allQ = qn.Predict(sess, featuresNP)
            next, nextDocId, reward = self.GetNextState(action, visited, unvisited, docsVisited)
            totReward += reward
            visited.add(next)
            unvisited.RemoveLink(next)
            docsVisited.add(nextDocId)

            alignedStr = ""
            nextNode = self.nodes[next]
            if nextNode.aligned > 0:
                alignedStr = "*"
                numAligned += 1

            if printQ:
                debugStr += "   " + str(curr) + "->" + str(next) + " " \
                         + str(action) + " " + str(allQ) + " " \
                         + unvisitedStr + " " \
                         + str(featuresNP) + " " \
                         + "\n"

            #print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
            mainStr += str(next) + alignedStr + "(" + str(reward) + ")->"
            curr = next

            if next == 0: break

            i += 1

        mainStr += " " + str(totReward)

        if printQ:
            print(debugStr, end="")
        print(mainStr)

        return numAligned


    def WalkAll(self, params, sess, qn):
        for start in range(self.ns):
            self.Walk(start, params, sess, qn, False)

    def GetNumberAligned(self, path):
        ret = 0
        for transition in path:
            next = transition.next
            nextNode = self.nodes[next]
            if nextNode.aligned > 0:
                ret += 1
        return ret


    def Neural(self, epoch, curr, params, sess, qnA, qnB, visited, unvisited, docsVisited):
        assert(curr != 0)
        #DEBUG = False
        #if curr == 31: DEBUG = True

        #print("curr", curr)
        unvisited.AddLinks(self, curr, visited, params)
        featuresNP = unvisited.GetFeaturesNP(self, params)
        nextStates = unvisited.GetNextStates(params)
        #print("   childIds", childIds, unvisited)

        action, Qs = qnA.Predict(sess, featuresNP)
        if np.random.rand(1) < params.eps:
            #if DEBUG: print("   random")
            action = np.random.randint(0, params.NUM_ACTIONS)
        
        next, nextDocId, r = self.GetNextState(action, visited, unvisited, docsVisited)
        #if DEBUG: print("   action", action, next, Qs)

        visited.add(next)
        unvisited.RemoveLink(next)
        nextUnvisited = unvisited.copy()
        docsVisited.add(nextDocId)

        if next == 0:
            done = True
            maxNextQ = 0.0
        else:
            assert(next != 0)
            done = False

            # Obtain the Q' values by feeding the new state through our network
            nextUnvisited.AddLinks(self, next, visited, params)
            nextFeaturesNP = nextUnvisited.GetFeaturesNP(self, params)
            nextAction, nextQs = qnA.Predict(sess, nextFeaturesNP)        
            #print("  nextAction", nextAction, nextQ)

            #assert(qnB == None)
            #maxNextQ = np.max(nextQs)

            _, nextQsB = qnB.Predict(sess, nextFeaturesNP)        
            maxNextQ = nextQsB[0, nextAction]
            
        targetQ = Qs
        #targetQ = np.array(Qs, copy=True)
        #print("  targetQ", targetQ)
        newVal = r + params.gamma * maxNextQ
        targetQ[0, action] = (1 - params.alpha) * targetQ[0, action] + params.alpha * newVal
        #targetQ[0, action] = newVal
        self.ZeroOutStop(targetQ, nextStates)

        #if DEBUG: print("   nextStates", nextStates)
        #if DEBUG: print("   targetQ", targetQ)

        transition = self.Transition(curr, next, done, np.array(featuresNP, copy=True), np.array(targetQ, copy=True))
        return transition

    def Trajectory(self, epoch, curr, params, sess, qns):
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

            transition = self.Neural(epoch, curr, params, sess, qnA, qnB, visited, unvisited, docsVisited)
            
            qnA.corpus.AddTransition(transition)

            curr = transition.next
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

class Node:
    def __init__(self, sqlconn, id, urlId, docId, lang, url):
        self.Link = namedtuple("Link", "text textLang parentNode childNode")

        self.id = id
        self.urlId = urlId
        self.docId = docId
        self.lang = lang
        self.url = url
        self.links = []
        self.aligned = 0

        self.CreateAlign(sqlconn)

    def CreateAlign(self, sqlconn):
        if self.docId is not None:
            sql = "select document1, document2 from document_align where document1 = %s or document2 = %s"
            val = (self.docId,self.docId)
            #print("sql", sql)
            sqlconn.mycursor.execute(sql, val)
            res = sqlconn.mycursor.fetchall()
            #print("aligned",  self.url, self.docId, res)

            if len(res) > 0:
                rec = res[0]
                if self.docId == rec[0]:
                    self.aligned = rec[1]
                elif self.docId == rec[1]:
                    self.aligned = rec[0]
                else:
                    assert(True)

        #print(self.Debug())

    def Debug(self):
        strLinks = ""
        for link in self.links:
            #strLinks += str(link.parentNode.id) + "->" + str(link.childNode.id) + " "
            strLinks += str(link.childNode.id) + " "

        return " ".join([str(self.id), str(self.urlId), 
                        StrNone(self.docId), StrNone(self.lang), 
                        str(self.aligned), self.url,
                        "links=", str(len(self.links)), ":", strLinks ] )

    def CreateLinks(self, sqlconn, env):
        #sql = "select id, text, url_id from link where document_id = %s"
        sql = "select link.id, link.text, link.text_lang, link.url_id, url.val from link, url where url.id = link.url_id and link.document_id = %s"
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

            if urlId in env.urlId2nodeId:
                nodeId = env.GetNodeIdFromURLId(urlId)
                childNode = env.nodes[nodeId]
                #print("child", self.docId, childNode.Debug())
            else:
                continue
                #id = len(nodes)
                #childNode = Node(sqlconn, id, urlId, None, None, url)
                #nodes[childNode.urlId] = childNode
                #nodesbyURL[childNode.url] = childNode
                #nodes.append(childNode)

            self.CreateLink(text, textLang, childNode)

    def CreateLink(self, text, textLang, childNode):            
        link = self.Link(text, textLang, self, childNode)
        self.links.append(link)

    def GetLinks(self, visited, params):
        ret = []
        for link in self.links:
            childNode = link.childNode
            childNodeId = childNode.id
            #print("   ", childNode.Debug())
            if childNodeId != self.id and childNodeId not in visited:
                ret.append(link)
        #print("   childIds", childIds)

        return ret

######################################################################################

class Candidates:
    def __init__(self):
        self.dict = {} # nodeid -> link
        self.vec = []

        self.dict[0] = []
        self.vec.append(0)

    def AddLink(self, link):
        childId = link.childNode.id
        if childId not in self.dict:
            self.dict[childId] = []
            self.vec.append(childId)
        self.dict[childId].append(link)

    def RemoveLink(self, childId):
        del self.dict[childId]
        self.vec.remove(childId)                

    def copy(self):
        ret = Candidates()
        ret.dict = self.dict.copy()
        ret.vec = self.vec.copy()

        return ret

    def AddLinks(self, env, curr, visited, params):
        currNode = env.nodes[curr]
        #print("   currNode", curr, currNode.Debug())
        newLinks = currNode.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for childId, links in self.dict.items():
            ret[0, i] = childId

            i += 1
            if i >= params.NUM_ACTIONS:
                #print("overloaded", len(self.dict), self.dict)
                break

        return ret


    def GetFeaturesNP(self, env, params):
        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for childId in self.vec:
            #ret[0, i] = childId
            links = self.dict[childId]

            if len(links) > 0:
                link = links[0]
                langId = env.GetLangId(link.textLang)
                ret[0, i] = langId

            i += 1
            if i >= params.NUM_ACTIONS:
                #print("overloaded", len(self.dict), self.dict)
                break

        return ret

    def GetNextState(self, action):
        if action >= len(self.vec):
            ret = 0
        else:
            ret = self.vec[action]
        return ret

    def GetNextStates(self, params):
        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for childId in self.vec:
            ret[0, i] = childId
            i += 1
            if i >= params.NUM_ACTIONS:
                break


        return ret

######################################################################################

def Train(params, env, sess, qns):
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        #startState = np.random.randint(0, env.ns)  # random start state
        #startState = 30
        startState = env.startNodeId
        #print("startState", startState)
        
        env.Trajectory(epoch, startState, params, sess, qns)
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)

        if epoch > 0 and epoch % params.walk == 0:
            #qns.q[0].PrintAllQ(params, env, sess)
            qns.q[0].PrintQ(0, params, env, sess)
            qns.q[0].PrintQ(31, params, env, sess)
            print()

            numAligned = env.Walk(startState, params, sess, qns.q[0], True)
            print("epoch", epoch, "loss", qns.q[0].corpus.losses[-1], "eps", params.eps, "alpha", params.alpha)
            print()

            #numAligned = env.GetNumberAligned(path)
            #print("path", numAligned, env.numAligned)
            if numAligned >= env.numAligned - 5:
                #print("got them all!")
                #eps = 1. / ((i/50) + 10)
                params.eps *= .99
                params.eps = max(0.1, params.eps)
                
                #params.alpha *= 0.99
                #params.alpha = max(0.3, params.alpha)
                
    # LAST BATCH
            
######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    # =============================================================
    sqlconn = MySQL()
    #siteMap = Sitemap(sqlconn, "www.visitbritain.com")
    # =============================================================
    env = Env(sqlconn, "www.vade-retro.fr/")

    params = LearningParams()

    tf.reset_default_graph()
    qns = Qnets(params, env)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        qns.q[0].PrintAllQ(params, env, sess)
        #env.WalkAll(params, sess, qn)
        print()

        Train(params, env, sess, qns)
        print("Trained")
        
        #qn.PrintAllQ(params, env, sess)
        #env.WalkAll(params, sess, qn)

        startState = env.startNodeId
        env.Walk(startState, params, sess, qns.q[0], True)

        plt.plot(qns.q[0].corpus.losses)
        plt.plot(qns.q[1].corpus.losses)
        #plt.show()

        plt.plot(qns.q[0].corpus.sumWeights)
        plt.plot(qns.q[1].corpus.sumWeights)
        #plt.show()

    print("Finished")


if __name__ == "__main__":
    Main()
