#!/usr/bin/env python3
from features3 import Env, MySQL
import numpy as np
import argparse

from common import MySQL


def crawl(env, lang='en'):
    for node_id in env.nodes:
        for link in env.nodes[node_id].links:
            print(link.text)


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

    crawl(env)

main()