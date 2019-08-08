#!/usr/bin/env python3
import numpy as np
import argparse
import hashlib

from common import MySQL
from helpers import Env


def fetchUrlLang(sqlconn, lang):
    pass


def getUrls(sqlconn, url):
    c = hashlib.md5()
    c.update(url.lower().encode())
    hashURL = c.hexdigest()

    sql = "SELECT val FROM url WHERE val LIKE %s"
    val = ('%' + url + '%',)
    sqlconn.mycursor.execute(sql, val)
    return sqlconn.mycursor.fetchall()


def populateGraph():
    pass


def crawl(sqlconn, env, lang='en'):
    for node in env.nodes.values():
        for url in getUrls(sqlconn, node.url):
            # Create Node for each fetched URL.
            print(url)
            if url:
                rootNode = env.CreateNode(sqlconn, {}, {}, node.urlId, url)

        # print(node.url)
        # print(node.lang)
        # for link in node.links:
        #     print(link.childNode.url)
        #     print(link.childNode.lang)


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
