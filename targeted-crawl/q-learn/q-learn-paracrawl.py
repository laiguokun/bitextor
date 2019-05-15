#!/usr/bin/env python3

import numpy as np
import pylab as plt
import random
import mysql.connector
import tensorflow as tf
from collections import namedtuple

######################################################################################
def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)

######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 1 #0.99
        self.lrn_rate = 0.1
        self.q_lrn_rate = 1
        self.max_epochs = 1 #00001
        self.eps = 1  # 0.7
        self.maxBatchSize = 3
        self.debug = False
        self.walk = 1000

        self.NUM_ACTIONS = 20

######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        # These lines establish the feed-forward part of the network used to choose actions
        EMBED_DIM = 80

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
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=params.lrn_rate)
        self.trainer = tf.train.AdamOptimizer() #learning_rate=params.lrn_rate)
        
        self.updateModel = self.trainer.minimize(self.loss)

        # server-side lookups
        self.lang2Id = {}

    def GetLangId(self, langStr):
        if langStr in self.lang2Id:
            ret = self.lang2Id[langStr]
        else:
            ret = len(self.lang2Id) + 1
            self.lang2Id[langStr] = ret

        return ret

    def PrintQ(self, curr, env, sess):
        # print("hh", next, hh)
        neighbours = env.GetNeighBours(curr)
        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: neighbours})
        print("curr=", curr, "a=", a, "allQ=", allQ, neighbours)

    def PrintAllQ(self, env, sess):
        for curr in range(env.ns):
            self.PrintQ(curr, env, sess)

######################################################################################
# helpers
class Env:
    def __init__(self):
        self.ns = 16  # number of states

######################################################################################

######################################################################################
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

class Sitemap:
    def __init__(self, sqlconn, url):
        # all nodes with docs
        sql = "select url.id, url.document_id, document.lang, url.val from url, document where url.document_id = document.id and val like %s"
        val = (url + "%",)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        self.nodes = {} # indexed by URL id
        self.nodesbyURL = {} # indexed by URL
        for rec in res:
            #print("rec", rec[0], rec[1])
            node = Node(sqlconn, rec[0], rec[1], rec[2], rec[3])
            self.nodes[node.urlId] = node
            self.nodesbyURL[node.url] = node
        #print("nodes", len(self.nodes))

        self.nodesWithDoc = self.nodes.copy()
        print("nodesWithDoc", len(self.nodesWithDoc))

        # links between nodes, possibly to nodes without doc
        for node in self.nodesWithDoc.values():
            node.CreateLinks(sqlconn, self.nodes, self.nodesbyURL)
            #print("node", node.Debug())
        print("all nodes", len(self.nodes))

        # lang id
        self.langIds = {}

        # print out
        #for node in self.nodes.values():
        #    print("node", node.Debug())

        #node = Node(sqlconn, url, True)
        #print("node", node.docId, node.urlId)       

    def GetLangId(self, langStr):
        if langStr in self.langIds:
            langId = self.langIds[langStr]
        else:
            langId = len(self.langIds)
            self.langIds[langStr] = langId
        return langId

    def GetRandomNode(self):
        l = list(self.nodesWithDoc.values())
        node = random.choice(l)
        return node 

    def GetNode(self, url):
        node = self.nodesbyURL[url]
        return node 


class Node:
    def __init__(self, sqlconn, urlId, docId, lang, url):
        self.urlId = urlId
        self.docId = docId
        self.lang = lang
        self.url = url
        self.links = []
        self.aligned = False

        if self.docId is not None:
            sql = "select * from document_align where document1 = %s or document2 = %s"
            val = (self.docId,self.docId)
            #print("sql", sql)
            sqlconn.mycursor.execute(sql, val)
            res = sqlconn.mycursor.fetchall()
            #print("aligned",  self.url, self.docId, res)

            if len(res) > 0:
                self.aligned = True

        print(self.Debug())

    def Debug(self):
        return " ".join([StrNone(self.urlId), StrNone(self.docId), StrNone(self.lang), str(len(self.links)), str(self.aligned), self.url])

    def CreateLinks(self, sqlconn, nodes, nodesbyURL):
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

            if urlId in nodes:
                childNode = nodes[urlId]
                #print("child", self.docId, childNode.Debug())
            else:
                childNode = Node(sqlconn, urlId, None, None, url)
                nodes[childNode.urlId] = childNode
                nodesbyURL[childNode.url] = childNode

            Link = namedtuple("Link", "text textLang parentNode childNode")
            link = Link(text, textLang, self, childNode)
            self.links.append(link)

    def GetUnvisitedLinks(self, visited):
        links = []
        for link in self.links:
            childNode = link.childNode
            if childNode.docId is not None and childNode.urlId not in visited:
                links.append(link)
        return links

class Corpus:
    def __init__(self):
        self.transitions = []

    def AddPath(self, path):
        for transition in path:
            self.transitions.append(transition)

    def GetBatch(self, maxBatchSize):
        ret = self.transitions[0:maxBatchSize]
        self.transitions = self.transitions[maxBatchSize:]
        return ret


def TrainSitemap(params, sitemap, sess, qn):
    losses = []
    sumWeights = []

    corpus = Corpus()

    for epoch in range(params.max_epochs):
        startState = sitemap.GetRandomNode() # random start state
        #startState = sitemap.GetNode("www.vade-retro.fr/")

        path = TrajectorySitemap(epoch, startState, params, sitemap, sess, qn)
        corpus.AddPath(path)

        while len(corpus.transitions) >= params.maxBatchSize:
            batch = corpus.GetBatch(params.maxBatchSize)
            UpdateQNSitemap(params, sitemap, sess, qn, batch)

def UpdateQNSitemap(params, sitemap, sess, qn, batch):
    batchSize = len(batch)
    #print("batchSize", batchSize)

    neighbours = np.zeros([batchSize, params.NUM_ACTIONS])
    targetQ = np.zeros([batchSize, params.NUM_ACTIONS])

    row = 0
    for transition in batch:
        link = transition.link
        parentNode = link.parentNode
        childNode = link.childNode
        print("transition", transition.targetQ, link.text, link.textLang, parentNode.urlId, "->", childNode.urlId, childNode.url)
        
        neighbours[row, 4] = qn.GetLangId(link.textLang)
        targetQ[row, :] = transition.targetQ

        row += 1

    print("   neighbours", neighbours)
    print("   targetQ", targetQ)

    outs = [qn.updateModel, qn.loss, qn.sumWeight, qn.Wout, qn.Whidden2, qn.BiasHidden2, qn.Qout, qn.embeddings, qn.embedding]
    _, loss, sumWeight, Wout, Whidden, BiasHidden, Qout, embeddings, embedding = sess.run(outs,
                                                                    feed_dict={qn.input: neighbours,
                                                                                qn.nextQ: targetQ})

def CalcQ(unvisitedLinks, params, sess, qn):
    # calc Q-value of next node
    assert(len(unvisitedLinks) <= params.NUM_ACTIONS)

    input = np.zeros([1, params.NUM_ACTIONS])

    col = 0
    for link in unvisitedLinks:
        input[0, col] = qn.GetLangId(link.textLang)

        col += 1

    #print("input", input)
    action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.input: input})
    action = action[0]

    return action, allQ

def TrajectorySitemap(epoch, curr, params, sitemap, sess, qn):
    Transition = namedtuple("Transition", "link targetQ")
    path = []
    visited = set()
    candidates = {}

    while True:
        print("curr", curr.Debug())
        visited.add(curr.urlId)

        unvisitedLinks = curr.GetUnvisitedLinks(visited)
        #print("  unvisitedLinks", len(unvisitedLinks))
        
        if len(unvisitedLinks) ==0:
            break

        action, allQ = CalcQ(unvisitedLinks, params, sess, qn)

        if np.random.rand(1) < params.eps:
            action = np.random.randint(0, 5)
        print("   action", action, len(unvisitedLinks))

        if action >= len(unvisitedLinks):
            # STOP
            maxQ1 = 0
            print("STOP")
        else:
            link = unvisitedLinks[action]
            nextNode = link.childNode

            nextVisited = visited.copy()
            nextVisited.add(nextNode.urlId)

            nextChildren = curr.GetUnvisitedLinks(nextVisited)
            nextAction, nextAllQ = CalcQ(nextChildren, params, sess, qn)

            maxQ1 = np.max(nextAllQ)
            print("   maxQ", nextNode.urlId, maxQ1)


        if nextNode.aligned:
            reward = 8.5
        else:
            reward = -1.0

        targetQ = np.array(allQ, copy=True)
        targetQ[0, action] = reward + params.gamma * maxQ1
        print("   targetQ", targetQ)

        transition = Transition(link, targetQ)
        path.append(transition)

        if action >= len(unvisitedLinks):
            break

        curr = nextNode

    print("path", curr.Debug(), len(path))
    return path

    
######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # =============================================================
    sqlconn = MySQL()
    #siteMap = Sitemap(sqlconn, "www.visitbritain.com")
    siteMap = Sitemap(sqlconn, "www.vade-retro.fr/")
    
    # =============================================================
    env = Env()

    params = LearningParams()

    tf.reset_default_graph()
    qn = Qnetwork(params, env)
    init = tf.global_variables_initializer()
    print("qn.Qout", qn.Qout)

    with tf.Session() as sess:
        sess.run(init)

        TrainSitemap(params, siteMap, sess, qn)
        print("Trained")
        exit()

        qn.PrintAllQ(env, sess)
        env.WalkAll(sess, qn)

        plt.plot(losses)
        plt.show()

        plt.plot(sumWeights)
        plt.show()

    print("Finished")


if __name__ == "__main__":
    Main()
