#!/usr/bin/env python3

import numpy as np
import pylab as plt
import tensorflow as tf

######################################################################################
class Qnetwork():
    def __init__(self):
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.placeholder(shape=[1, 15], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([15, 5], 0, 0.01))
        self.Qout = tf.matmul(self.inputs1, self.W)
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

    def GetNextState(self, curr, action):
        if action == 0:
            next = curr - 5
        elif action == 1:
            next = curr + 1
        elif action == 2:
            next = curr + 5
        elif action == 3:
            next = curr - 1
        elif action == 4:
            next = curr
        #assert(next >= 0)
        #print("next", next)

        die = False
        if action == 4:
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


def get_rnd_next_state(s, env):
    actions = env.get_poss_next_actions(s)

    i = np.random.randint(0, len(actions))
    action = actions[i]
    next_state, reward, die = env.GetNextState(s, action)

    return next_state, action, reward, die


def my_print(Q):
    rows = len(Q);
    cols = len(Q[0])
    print("       0      1      2      3      4")
    for i in range(rows):
        print("%d " % i, end="")
        if i < 10: print(" ", end="")
        for j in range(cols): print(" %6.2f" % Q[i, j], end="")
        print("")
    print("")


def Walk(start, Q, env):
    curr = start
    i = 0
    totReward = 0
    print(str(curr) + "->", end="")
    while True:
        #print("curr", curr)
        action = np.argmax(Q[curr])
        next, reward, die = env.GetNextState(curr, action)
        totReward += reward

        print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
        #print(str(next) + "->", end="")
        curr = next

        if die: break
        if curr == env.goal: break

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("done", totReward)


######################################################################################
def GetMaxQ(next_s, actions, Q, env):
    #if actions == 4:
    #    return 0

    max_Q = -9999.99
    for j in range(len(actions)):
        nn_a = actions[j]
        nn_s = env.GetNextState(next_s, nn_a)

        q = Q[next_s, nn_a]
        if q > max_Q:
            max_Q = q
    return max_Q

def Tabular(curr_s, Q, gamma, lrn_rate, env):
    next_s, action, reward, die = get_rnd_next_state(curr_s, env)
    actions = env.get_poss_next_actions(next_s)

    DEBUG = False
    # DEBUG = action == 4
    # DEBUG = curr_s == 0

    max_Q = GetMaxQ(next_s, actions, Q, env)

    if DEBUG:
        print("max_Q", max_Q)
        before = Q[curr_s][action]

    prevQ = ((1 - lrn_rate) * Q[curr_s][action])
    V = lrn_rate * (reward + (gamma * max_Q))
    Q[curr_s][action] = prevQ + V

    if DEBUG:
        after = Q[curr_s][action]
        print("Q", curr_s, reward, before, after)

    if die or curr_s == env.goal:
        return next_s, True

    return next_s, False

def Neural(curr_s, gamma, lrn_rate, env, sess, qn):
    # NEURAL
    curr_1Hot = np.identity(env.ns)[curr_s:curr_s + 1]
    # print("hh", next_s, hh)
    a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs1: curr_1Hot})
    a = a[0]
    next_s, r, die = env.GetNextState(curr_s, a)
    #print("curr_s=", curr_s, "a=", a, "allQ=", allQ)

    # Obtain the Q' values by feeding the new state through our network
    next1Hot = np.identity(env.ns)[next_s:next_s + 1]
    # print("  hh2", hh2)
    Q1 = sess.run(qn.Qout, feed_dict={qn.inputs1: next1Hot})
    # print("  Q1", Q1)
    maxQ1 = np.max(Q1)
    # print("  Q1", Q1, maxQ1)

    targetQ = allQ
    targetQ[0, a] = r + gamma * maxQ1

    inputs = np.identity(env.ns)[curr_s: curr_s + 1]
    _, W1 = sess.run([qn.updateModel, qn.W], feed_dict={qn.inputs1: inputs, qn.nextQ: targetQ})

    return next_s, die


def Trajectory(curr_s, Q, gamma, lrn_rate, env, sess, qn):
    while (True):
        next_s, done = Neural(curr_s, gamma, lrn_rate, env, sess, qn)
        #next_s, done = Tabular(curr_s, Q, gamma, lrn_rate, env)
        curr_s = next_s

        if done: break
    #print()

    if (np.max(Q) > 0):
        score = (np.sum(Q / np.max(Q) * 100))
    else:
        score = (0)

    return score

def Train(Q, gamma, lrn_rate, max_epochs, env):
    tf.reset_default_graph()
    qn = Qnetwork()
    init = tf.initialize_all_variables()
    print("qn.Qout", qn.Qout)

    with tf.Session() as sess:
        sess.run(init)

        scores = []

        for i in range(0, max_epochs):
            curr_s = np.random.randint(0, env.ns)  # random start state
            score = Trajectory(curr_s, Q, gamma, lrn_rate, env, sess, qn)
            scores.append(score)

    return scores


######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    print("Setting up maze in memory")

    # R = np.random.rand(15, 15)  # Rewards
    MOVE_REWARD = 0

    # =============================================================

    Q = np.empty(shape=[15, 5], dtype=np.float)  # Quality
    Q[:] = 0

    print("Analyzing maze with RL Q-learning")
    start = 0;
    gamma = 0.99
    lrn_rate = 0.5
    max_epochs = 10000
    env = Env()

    scores = Train(Q, gamma, lrn_rate, max_epochs, env)
    print("Trained")

    print("The Q matrix is: \n ")
    my_print(Q)

    #
    # plt.plot(scores)
    #plt.show()

    #print("Using Q to go from 0 to goal (14)")
    #Walk(start, goal, Q)

    for start in range(0,env.ns):
        Walk(start, Q, env)

    print("Finished")


if __name__ == "__main__":
    Main()
