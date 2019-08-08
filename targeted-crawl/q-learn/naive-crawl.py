#!/usr/bin/env python3
import numpy as np
import argparse
import hashlib

from common import MySQL
from helpers import Env

def crawl(sqlconn, env, lang='en'):
    todo = []
    todo.append(env.rootNode)

    visited = set()

    while len(todo) > 0:
        node = todo.pop()

        if node.urlId not in visited:
            print("node", node.Debug())
            visited.add(node.urlId)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

def main():
    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.)")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    sqlconn = MySQL(options.configFile)

    hostName = "http://vade-retro.fr/"
    env = Env(sqlconn, hostName)

    crawl(sqlconn, env)

main()
