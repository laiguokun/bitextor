#!/usr/bin/env python3

import numpy as np
import pylab as plt


######################################################################################
# helpers
def get_poss_next_states(s, F, ns):
    poss_next_states = []
    for j in range(ns):
        if F[s, j] == 1: poss_next_states.append(j)
    return poss_next_states


def get_rnd_next_state(s, F, ns):
    poss_next_states = get_poss_next_states(s, F, ns)
    next_state = \
        poss_next_states[np.random.randint(0, \
                                           len(poss_next_states))]
    return next_state


def my_print(Q):
    rows = len(Q);
    cols = len(Q[0])
    print("       0      1      2      3      4      5\
      6      7      8      9      10     11     12\
     13     14")
    for i in range(rows):
        print("%d " % i, end="")
        if i < 10: print(" ", end="")
        for j in range(cols): print(" %6.2f" % Q[i, j], end="")
        print("")
    print("")


def walk(start, goal, Q):
    curr = start
    i = 0
    print(str(curr) + "->", end="")
    while curr != goal:
        next = np.argmax(Q[curr])
        print(str(next) + "->", end="")
        curr = next

        i += 1
        if i > 50:
            print("LOOPING")
            break

    print("done")


######################################################################################

def train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
    scores = []

    for i in range(0, max_epochs):
        curr_s = np.random.randint(0, ns)  # random start state

        while (True):
            next_s = get_rnd_next_state(curr_s, F, ns)
            poss_next_next_states = \
                get_poss_next_states(next_s, F, ns)

            max_Q = -9999.99
            for j in range(len(poss_next_next_states)):
                nn_s = poss_next_next_states[j]
                q = Q[next_s, nn_s]
                if q > max_Q:
                    max_Q = q

            before = Q[curr_s][next_s]
            # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
            Q[curr_s][next_s] = ((1 - lrn_rate) * Q[curr_s] \
                [next_s]) + (lrn_rate * (R[curr_s][next_s] + \
                                         (gamma * max_Q)))
            after = Q[curr_s][next_s]
            # print("Q", before, after)

            curr_s = next_s
            if curr_s == goal: break

        if (np.max(Q) > 0):
            score = (np.sum(Q / np.max(Q) * 100))
        else:
            score = (0)

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
    MOVE_REWARD = 1
    R = np.zeros(shape=[15, 15], dtype=np.float)  # Rewards
    R[0, 1] = MOVE_REWARD;
    R[0, 5] = MOVE_REWARD;
    R[1, 0] = MOVE_REWARD;
    R[2, 3] = MOVE_REWARD
    R[3, 2] = MOVE_REWARD;
    R[3, 4] = MOVE_REWARD;
    R[3, 8] = MOVE_REWARD;
    R[4, 3] = MOVE_REWARD
    R[4, 9] = MOVE_REWARD;
    R[5, 0] = MOVE_REWARD;
    R[5, 6] = MOVE_REWARD;
    R[5, 10] = MOVE_REWARD
    R[6, 5] = MOVE_REWARD;
    R[7, 8] = MOVE_REWARD;
    R[7, 12] = MOVE_REWARD;
    R[8, 3] = MOVE_REWARD
    R[8, 7] = MOVE_REWARD;
    R[9, 4] = MOVE_REWARD;
    R[9, 14] = 10.0;  # final move
    R[10, 5] = MOVE_REWARD
    R[10, 11] = MOVE_REWARD;
    R[11, 10] = MOVE_REWARD;
    R[11, 12] = MOVE_REWARD
    R[12, 7] = MOVE_REWARD;
    R[12, 11] = MOVE_REWARD;
    R[12, 13] = MOVE_REWARD
    R[13, 12] = MOVE_REWARD;
    R[14, 14] = MOVE_REWARD
    print("R", R)

    # =============================================================

    Q = np.zeros(shape=[15, 15], dtype=np.float32)  # Quality

    print("Analyzing maze with RL Q-learning")
    start = 0;
    goal = 14
    ns = 15  # number of states
    gamma = 0.5
    lrn_rate = 0.5
    max_epochs = 10
    scores = train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs)
    print("Trained")

    print("The Q matrix is: \n ")
    my_print(Q)

    #plt.plot(scores)
    #plt.show()

    print("Using Q to go from 0 to goal (14)")
    walk(start, goal, Q)

    # for s in range(0,10):
    #    nextStates = get_poss_next_states(s, F, ns)
    #    print("nextStates", nextStates)

    #    nextState = get_rnd_next_state(s, F, ns)
    #    print("   nextState", nextState)

    print("Finished")


if __name__ == "__main__":
    Main()
