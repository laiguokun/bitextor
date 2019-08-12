#!/usr/bin/env python3
import numpy as np
import argparse
import hashlib

from common import MySQL
from helpers import Env


def naive(sqlconn, env, maxDocs):
    todo = []
    todo.append(env.rootNode)

    visited = set()

    while len(todo) > 0 and len(visited) < maxDocs:
        node = todo.pop()

        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

    numParallelDocs = NumParallelDocs(env, visited)
    print("numParallelDocs", len(visited), numParallelDocs)

def NumParallelDocs(env, visited):
    ret = 0
    for urlId in visited:
        node = env.nodes[urlId]
        #print("node", node.Debug())

        if node.alignedNode is not None and node.alignedNode.urlId in visited:
            ret += 1

    return ret

def main():
    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.)")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    sqlconn = MySQL(options.configFile)

    hostName = "http://vade-retro.fr/"
    #hostName = "http://www.buchmann.ch/"
    env = Env(sqlconn, hostName)

    #crawl_method1(sqlconn, env)
    naive(sqlconn, env, 20)

main()
