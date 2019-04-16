#!/usr/bin/env python3

import numpy as np
import pylab as plt
import tensorflow as tf
import mysql.connector

######################################################################################
class Qnetwork():
    def __init__(self):
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.placeholder(shape=[1, 15], dtype=tf.float32)
        self.hidden = self.inputs1

        #self.Whidden = tf.Variable(tf.random_uniform([15, 15], 0, 0.01))
        #self.hidden = tf.matmul(self.inputs1, self.Whidden)

        #self.biashidden = tf.Variable(tf.random_uniform([1, 15], 0, 0.01))
        #self.hidden = tf.add(self.hidden, self.biashidden)

        #self.hidden = tf.nn.relu(self.hidden)
        #self.hidden = tf.nn.softmax(self.hidden)

        self.W = tf.Variable(tf.random_uniform([15, 5], 0, 0.01))

        #self.Qout = tf.matmul(self.inputs1, self.W)
        self.Qout = tf.matmul(self.hidden, self.W)
        #self.Qout = tf.nn.softmax(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 5], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = self.trainer.minimize(self.loss)

######################################################################################
# helpers
class Env:
    def __init__(self):
        self.goal = 14
        self.ns = 15  # number of states

        self.F = np.zeros(shape=[15, 15], dtype=np.int)  # Feasible
        self.F[0, 1] = 1;
        self.F[0, 5] = 1;
        self.F[1, 0] = 1;
        self.F[2, 3] = 1;
        self.F[3, 2] = 1
        self.F[3, 4] = 1;
        self.F[3, 8] = 1;
        self.F[4, 3] = 1;
        self.F[4, 9] = 1;
        self.F[5, 0] = 1
        self.F[5, 6] = 1;
        self.F[5, 10] = 1;
        self.F[6, 5] = 1;
        # self.F[6, 7] = 1; # hole
        # self.F[7, 6] = 1; # hole
        self.F[7, 8] = 1;
        self.F[7, 12] = 1
        self.F[8, 3] = 1;
        self.F[8, 7] = 1;
        self.F[9, 4] = 1;
        self.F[9, 14] = 1;
        self.F[10, 5] = 1
        self.F[10, 11] = 1;
        self.F[11, 10] = 1;
        self.F[11, 12] = 1;
        self.F[12, 7] = 1;
        self.F[12, 11] = 1;
        self.F[12, 13] = 1;
        self.F[13, 12] = 1;
        self.F[14, 14] = 1
        #print("F", self.F)

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

        self.matchedDocIds = self.GetMatches()
        print("matchedDocIds", self.matchedDocIds)
        startNode = self.GetStartNode("www.vade-retro.fr/")
        print("startNode", startNode)
        children = self.GetChildren(startNode)
        print("children", children)

        self.docsVisited = set()

    def GetNextState(self, curr, action):
        if action == 1:
            next = curr - 5
        elif action == 2:
            next = curr + 1
        elif action == 3:
            next = curr + 5
        elif action == 4:
            next = curr - 1
        elif action == 0:
            next = curr
        #assert(next >= 0)
        #print("next", next)

        die = False
        if action == 0:
            reward = 0
            die = True
        elif next < 0 or next >= self.ns or self.F[curr, next] == 0:
            next = curr
            reward = -100
            die = True
        elif next == self.goal:
            reward = 8.5
        else:
            reward = -1

        return next, reward, die

    def get_poss_next_actions(self, s):
        actions = []
        actions.append(0)
        actions.append(1)
        actions.append(2)
        actions.append(3)
        actions.append(4)

        #print("  actions", actions)
        return actions

    # paracrawl ##########################################################
    def GetMatches(self):
        sql = "select document1, document2 from document_align"
        self.mycursor.execute(sql)
        res = self.mycursor.fetchall()

        docIds = []
        for row in res:
            docIds.append(row[0])
            docIds.append(row[1])

        return docIds

    def GetStartNode(self, url):
        sql = "select id, document_id from url where val = %s"
        val = (url,)
        self.mycursor.execute(sql, val)
        res = self.mycursor.fetchone()
        assert (res is not None)
        docId = res[1]

        return docId

    def GetChildren(self, parentNode):
        sql = "select link.document_id, url.val, url.document_id " \
              + "from link, url " \
              + "where url.id = link.url_id " \
              + "and url.document_id is not null " \
              + "and link.document_id = %s"
        val = (parentNode,)
        self.mycursor.execute(sql, val)
        res = self.mycursor.fetchall()
        # print("  res", len(res))

        children = []
        for row in res:
            # print("  row", row)
            childNode = row[2]
            children.append(childNode)

        return children


######################################################################################
def Int2Arrray(num, size):
    ret = np.identity(size)[num:num + 1]
    return ret

    str = np.binary_repr(num).zfill(size)
    l = list(str)
    ret = np.array(l, ndmin=2).astype(np.float)
    #print("num", num, ret)
    return ret


def Neural(curr_s, eps, gamma, lrn_rate, env, sess, qn):
    # NEURAL
    #startNode = env.GetStartNode("www.vade-retro.fr/")
    #curr_1Hot = Int2Arrray(startNode, env.ns)

    curr_1Hot = Int2Arrray(curr_s, env.ns)
    #print("curr_s", curr_s, curr_1Hot)

    a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs1: curr_1Hot})
    #print("a", a, allQ)
    a = a[0]
    if np.random.rand(1) < eps:
        a = np.random.randint(0, 5)

    next_s, r, die = env.GetNextState(curr_s, a)
    #print("curr_s=", curr_s, "a=", a, "next_s=", next_s, "r=", r, "allQ=", allQ)

    # Obtain the Q' values by feeding the new state through our network
    next1Hot = Int2Arrray(next_s, env.ns)
    # print("  hh2", hh2)
    Q1 = sess.run(qn.Qout, feed_dict={qn.inputs1: next1Hot})
    #print("  Q1", Q1)

    maxQ1 = np.max(Q1)
    # print("  Q1", Q1, maxQ1)

    targetQ = allQ
    #print("  targetQ", targetQ)
    targetQ[0, a] = r + gamma * maxQ1
    #print("  targetQ", targetQ)

    inputs = Int2Arrray(curr_s, env.ns)
    _, W1 = sess.run([qn.updateModel, qn.W], feed_dict={qn.inputs1: inputs, qn.nextQ: targetQ})
    #print("  Whidden1", Whidden1)

    return next_s, die


def Trajectory(curr_s, eps, gamma, lrn_rate, env, sess, qn):
    while (True):
        next_s, done = Neural(curr_s, eps, gamma, lrn_rate, env, sess, qn)
        #next_s, done = Tabular(curr_s, Q, gamma, lrn_rate, env)
        curr_s = next_s

        if done: break
    #print()

def Train(eps, gamma, lrn_rate, max_epochs, env, sess, qn):

    scores = []

    for i in range(0, max_epochs):
        curr_s = np.random.randint(0, env.ns)  # random start state
        Trajectory(curr_s, eps, gamma, lrn_rate, env, sess, qn)

        #eps = 1. / ((i/50) + 10)
        #eps *= .99
        #print("eps", eps)

    return scores

######################################################################################

def my_print(env, sess, qn):
    for curr_s in range(15):
        curr_1Hot = Int2Arrray(curr_s, env.ns)
        # print("hh", next_s, hh)
        a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs1: curr_1Hot})
        print("curr_s=", curr_s, "a=", a, "allQ=", allQ)

def Walk(start, env, sess, qn):
    curr_s = start
    i = 0
    totReward = 0
    print(str(curr_s) + "->", end="")
    while True:
        # print("curr", curr)
        curr_1Hot = Int2Arrray(curr_s, env.ns)
        # print("hh", next_s, hh)
        action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs1: curr_1Hot})
        action= action[0]
        next, reward, die = env.GetNextState(curr_s, action)
        totReward += reward

        print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
        # print(str(next) + "->", end="")
        curr_s = next

        if die: break
        if curr_s == env.goal: break

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("done", totReward)

######################################################################################

######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Setting up maze in memory")

    # =============================================================


    # =============================================================
    print("Analyzing maze with RL Q-learning")
    start = 0;
    gamma = 0.99
    lrn_rate = 0.5
    max_epochs = 20000
    env = Env()
    eps = 1 #0.7

    tf.reset_default_graph()
    qn = Qnetwork()
    init = tf.initialize_all_variables()
    print("qn.Qout", qn.Qout)

    # =============================================================
    with tf.Session() as sess:
        sess.run(init)

        scores = Train(eps, gamma, lrn_rate, max_epochs, env, sess, qn)
        print("Trained")

        my_print(env, sess, qn)

        for start in range(0,env.ns):
            Walk(start, env, sess, qn)

        # plt.plot(scores)
        # plt.show()

        print("Finished")


if __name__ == "__main__":
    Main()
