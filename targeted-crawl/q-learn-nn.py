#!/usr/bin/env python3

import numpy as np
import pylab as plt


######################################################################################
# helpers
def GetNextState(curr, action):
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
    return next

def get_poss_next_actions(s, F, ns):
    #print("s", s)
    actions = []
    for j in range(ns):
        #print("s", s, j)
        if F[s, j] == 1:
            if s - 1 == j:
                actions.append(3)
            elif s == j - 1:
                actions.append(1)
            elif s < j:
                actions.append(2)
            elif s > j:
                actions.append(0)
            else:
                assert(s == j)
                actions.append(4)

    #print("  ", poss_next_states, actions)
    return actions


def get_rnd_next_state(s, F, ns):
    actions = get_poss_next_actions(s, F, ns)

    i = np.random.randint(0, len(actions))
    action = actions[i]
    next_state = GetNextState(s, action)

    return next_state, action


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


def Walk(start, goal, Q):
    curr = start
    i = 0
    print(str(curr) + "->", end="")
    while curr != goal:
        action = np.argmax(Q[curr])
        next = GetNextState(curr, action)

        print(str(next) + "->", end="")
        curr = next

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("done")


######################################################################################

def Trajectory(curr_s, F, R, Q, gamma, lrn_rate, goal, ns):
    while (True):
        next_s, action = get_rnd_next_state(curr_s, F, ns)
        actions = get_poss_next_actions(next_s, F, ns)

        max_Q = -9999.99
        for j in range(len(actions)):
            nn_a = actions[j]
            nn_s = GetNextState(next_s, nn_a)
            q = Q[next_s, nn_a]
            if q > max_Q:
                max_Q = q

        #before = Q[curr_s][next_s]
        # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
        prevQ = ((1 - lrn_rate) * Q[curr_s][action])
        V = (lrn_rate * (R[curr_s][next_s] + (gamma * max_Q)))
        Q[curr_s][action] = prevQ + V
        #after = Q[curr_s][next_s]
        # print("Q", before, after)

        curr_s = next_s
        if curr_s == goal: break

    if (np.max(Q) > 0):
        score = (np.sum(Q / np.max(Q) * 100))
    else:
        score = (0)

    return score

def Train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
    scores = []

    for i in range(0, max_epochs):
        curr_s = np.random.randint(0, ns)  # random start state
        score = Trajectory(curr_s, F, R, Q, gamma, lrn_rate, goal, ns)
        scores.append(score)

    return scores


######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    print("Setting up maze in memory")

    F = np.zeros(shape=[15, 15], dtype=np.int)  # Feasible
    F[0, 1] = 1;
    F[0, 5] = 1;
    F[1, 0] = 1;
    F[2, 3] = 1;
    F[3, 2] = 1
    F[3, 4] = 1;
    F[3, 8] = 1;
    F[4, 3] = 1;
    F[4, 9] = 1;
    F[5, 0] = 1
    F[5, 6] = 1;
    F[5, 10] = 1;
    F[6, 5] = 1;
    #F[6, 7] = 1; # hole
    #F[7, 6] = 1; # hole
    F[7, 8] = 1;
    F[7, 12] = 1
    F[8, 3] = 1;
    F[8, 7] = 1;
    F[9, 4] = 1;
    F[9, 14] = 1;
    F[10, 5] = 1
    F[10, 11] = 1;
    F[11, 10] = 1;
    F[11, 12] = 1;
    F[12, 7] = 1;
    F[12, 11] = 1;
    F[12, 13] = 1;
    F[13, 12] = 1;
    F[14, 14] = 1
    print("F", F)

    # R = np.random.rand(15, 15)  # Rewards
    MOVE_REWARD = 0
    R = np.empty(shape=F.shape, dtype=np.float)  # Rewards
    R[:] = 0
    for i in range(0, F.shape[0]):
        for j in range(0, F.shape[1]):
            if F[i, j] > 0:
                R[i, j] = MOVE_REWARD;

    R[:, 14] = 10.0;  # final move
    print("R", R)

    # =============================================================

    Q = np.empty(shape=[15, 5], dtype=np.float)  # Quality
    Q[:] = 0

    print("Analyzing maze with RL Q-learning")
    start = 0;
    goal = 14
    ns = 15  # number of states
    gamma = 0.5
    lrn_rate = 0.5
    max_epochs = 1000
    scores = Train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs)
    print("Trained")

    print("The Q matrix is: \n ")
    my_print(Q)

    #plt.plot(scores)
    #plt.show()

    print("Using Q to go from 0 to goal (14)")
    Walk(start, goal, Q)

    # for s in range(0,10):
    #    nextStates = get_poss_next_states(s, F, ns)
    #    print("nextStates", nextStates)

    #    nextState = get_rnd_next_state(s, F, ns)
    #    print("   nextState", nextState)

    print("Finished")


if __name__ == "__main__":
    Main()
