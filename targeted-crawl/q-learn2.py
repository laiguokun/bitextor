#!/usr/bin/env python3

import numpy as np
import pylab as plt
import networkx as nx


######################################################################################
# helper

def ShowGraph(points_list):
    G = nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()

def available_actions(state, R):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1] # only where rewards >= 0. Weird
    return av_act

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range,1))
    return next_action

######################################################################################
#train
def update(current_state, action, gamma, Q, R):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    Q[current_state, action] = R[current_state, action] + gamma * max_value
    #print('max_value', R[current_state, action] + gamma * max_value)

    if (np.max(Q) > 0):
        return (np.sum(Q / np.max(Q) * 100))
    else:
        return (0)

def Train(Q, R, gamma):
    # Training
    scores = []
    for i in range(700):
        current_state = np.random.randint(0, int(Q.shape[0]))
        available_act = available_actions(current_state, R)
        action = sample_next_action(available_act)
        score = update(current_state, action, gamma, Q, R)
        scores.append(score)
        #print ('Score:', str(score))


######################################################################################

def Main():
    print("Starting")
    np.random.seed()

    # map cell to cell, add circular cell to goal point
    points_list = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), (2, 3), (2, 7)]
    #ShowGraph(points_list)

    goal = 7

    # how many points in graph? x points
    MATRIX_SIZE = 8

    # create matrix x*y
    R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
    R *= -1

    # assign zeros to paths and 100 to goal-reaching point
    for point in points_list:
        print(point, point[::-1])
        if point[1] == goal:
            R[point] = 100
        else:
            R[point] = 0

        # reverse, undirected graphs
        if point[0] == goal:
            R[point[::-1]] = 100
        else:
            # reverse of point
            R[point[::-1]] = 0

    # add goal point round trip
    R[goal, goal] = 100
    Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

    print("R", R.shape, R)
    print("Q", Q.shape, Q)

    # learning parameter
    gamma = 0.8

    initial_state = 1

    #available_act = available_actions(initial_state, R)
    #print("available_act", available_act)
    #action = sample_next_action(available_act)
    #print("action", action)

    #update(initial_state, action, gamma, Q, R)

    Train(Q, R, gamma)
    print("R", R.shape, R)
    print("Q", Q.shape, Q)

    print("Trained Q matrix:")
    print(Q / np.max(Q) * 100)

    # Testing
    current_state = 0
    steps = [current_state]

    while current_state != 7:

        next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))
        else:
            next_step_index = int(next_step_index)

        steps.append(next_step_index)
        current_state = next_step_index

    print("Most efficient path:")
    print(steps)

    print("Finished")

if __name__ == "__main__":
    Main()

