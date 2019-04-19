#!/usr/bin/env python3

import numpy as np
import pylab as plt
import tensorflow as tf

######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 0.99
        self.lrn_rate = 0.1
        self.max_epochs = 500 #0 #0
        self.eps = 1  # 0.7

######################################################################################
class Qnetwork():
    def __init__(self, lrn_rate):
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[1, 15], dtype=tf.float32)
        self.hidden = self.inputs

        self.Whidden = tf.Variable(tf.random_uniform([15, 15], 0, 0.01))
        #self.Whidden = tf.nn.softmax(self.Whidden, axis=1)
        #self.Whidden = tf.contrib.layers.l2_regularizer(self.Whidden)

        #self.hidden = tf.matmul(self.hidden, self.Whidden)

        self.W = tf.Variable(tf.random_uniform([15, 5], 0, 0.01))

        #self.Qout = tf.matmul(self.inputs, self.W)
        self.Qout = tf.matmul(self.hidden, self.W)
        #self.Qout = tf.nn.softmax(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 5], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.updateModel = self.trainer.minimize(self.loss)

    def ResizeBatch(self, batchSize):
        tf.reshape(self.inputs, [batchSize, 15])
        tf.reshape(self.nextQ, [batchSize, 5])

    def UpdateQN(self, path, params, env, sess):
        batchSize = 1 #len(path)
        #print("path", batchSize)
        self.ResizeBatch(batchSize)

        inputs = np.empty([batchSize, 15])
        outputs = np.empty([batchSize, 15])
        targetQ = np.empty([batchSize, 5])

        for tuple in path:
            # print(tuple)
            curr = tuple[1]
            next = tuple[2]
            action = tuple[3]
            allQ = tuple[4]
            r = tuple[5]
            curr1Hot = Int2Arrray(curr, env.ns)
            #print("  curr_1Hot", curr_1Hot.shape, inputs.shape)
            inputs[0, :] = curr1Hot

            # Obtain the Q' values by feeding the new state through our network
            next1Hot = Int2Arrray(next, env.ns)
            #print("  next1Hot", type(next1Hot))
            outputs[0, :] = next1Hot

            Q1 = sess.run(self.Qout, feed_dict={self.inputs: next1Hot})
            # print("  Q1", Q1)

            maxQ1 = np.max(Q1)
            # print("  Q1", Q1, maxQ1)

            targetQ[0, :] = allQ
            #print("  targetQ", type(targetQ), targetQ.shape)
            targetQ[0, action] = r + params.gamma * maxQ1
            # print("  targetQ", targetQ)

            # _, W1 = sess.run([self.updateModel, self.W], feed_dict={self.inputs: inputs, self.nextQ: targetQ})

            _, W1, Whidden, hidden = sess.run([self.updateModel, self.W, self.Whidden, self.hidden],
                                              feed_dict={self.inputs: inputs, self.nextQ: targetQ})
            # print("  Whidden", Whidden, inputs.shape, Whidden.shape)
            # sumWhidden = np.sum(Whidden, 1)
            # sumhidden = np.sum(hidden)
            # print("sums", sumWhidden, sumhidden)
            # sdssess
            # if epoch % 10000 == 0:
            #    print("  Whidden", Whidden)


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

######################################################################################
def Int2Arrray(num, size):
    ret = np.identity(size)[num:num + 1]
    return ret

    str = np.binary_repr(num).zfill(size)
    l = list(str)
    ret = np.array(l, ndmin=2).astype(np.float)
    #print("num", num, ret)
    return ret


def Neural(epoch, curr, params, env, sess, qn):
    # NEURAL
    #startNode = env.GetStartNode("www.vade-retro.fr/")
    #curr_1Hot = Int2Arrray(startNode, env.ns)

    curr_1Hot = Int2Arrray(curr, env.ns)
    #print("curr", curr, curr_1Hot)

    action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs: curr_1Hot})
    #print("a", a, allQ)
    action = action[0]
    if np.random.rand(1) < params.eps:
        action = np.random.randint(0, 5)

    next, r, die = env.GetNextState(curr, action)
    #print("curr=", curr, "a=", a, "next=", next, "r=", r, "allQ=", allQ)

    return (die, curr, next, action, allQ, r)

def Trajectory(epoch, curr, params, env, sess, qn):
    path = []
    while (True):
        tuple = Neural(epoch, curr, params, env, sess, qn)
        path.append(tuple)

        curr = tuple[1]
        if tuple[0]: break

    #print(path)
    qn.UpdateQN(path, params, env, sess)

    return next

def Train(params, env, sess, qn):

    scores = []

    for epoch in range(params.max_epochs):
        curr = np.random.randint(0, env.ns)  # random start state
        stopState = Trajectory(epoch, curr, params, env, sess, qn)

        if stopState == env.goal:
            #params.eps = 1. / ((i/50) + 10)
            params.eps *= 1 #.999
            #print("eps", params.eps)

    return scores

######################################################################################

def my_print(env, sess, qn):
    for curr in range(15):
        curr_1Hot = Int2Arrray(curr, env.ns)
        # print("hh", next, hh)
        a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs: curr_1Hot})
        print("curr=", curr, "a=", a, "allQ=", allQ)

def Walk(start, env, sess, qn):
    curr = start
    i = 0
    totReward = 0
    print(str(curr) + "->", end="")
    while True:
        # print("curr", curr)
        curr_1Hot = Int2Arrray(curr, env.ns)
        # print("hh", next, hh)
        action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs: curr_1Hot})
        action= action[0]
        next, reward, die = env.GetNextState(curr, action)
        totReward += reward

        print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
        # print(str(next) + "->", end="")
        curr = next

        if die: break
        if curr == env.goal: break

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("done", totReward)

######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Setting up maze in memory")

    # =============================================================


    # =============================================================
    print("Analyzing maze with RL Q-learning")
    env = Env()
    params = LearningParams()

    tf.reset_default_graph()
    qn = Qnetwork(params.lrn_rate)
    init = tf.initialize_all_variables()

    # =============================================================
    with tf.Session() as sess:
        sess.run(init)

        scores = Train(params, env, sess, qn)
        print("Trained")

        my_print(env, sess, qn)

        for start in range(env.ns):
            Walk(start, env, sess, qn)

        # plt.plot(scores)
        # plt.show()

        print("Finished")


if __name__ == "__main__":
    Main()
