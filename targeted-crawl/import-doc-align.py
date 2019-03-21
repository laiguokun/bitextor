#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib


######################################################################################
def GetURLId(mycursor, url):
    c = hashlib.md5()
    c.update(url.encode())
    hashURL = c.hexdigest()

    sql = "SELECT id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    assert(res is not None)

    return res[0]

######################################################################################

print("Starting")

oparser = argparse.ArgumentParser(description="import-mysql")
oparser.add_argument('--lang1', dest='l1', help='Language l1 in the crawl', required=True)
oparser.add_argument('--lang2', dest='l2', help='Language l2 in the crawl', required=True)
options = oparser.parse_args()

mydb = mysql.connector.connect(
    host="localhost",
    user="paracrawl_user",
    passwd="paracrawl_password",
    database="paracrawl",
    charset='utf8'
)
mydb.autocommit = False
mycursor = mydb.cursor()

for line in sys.stdin:
    line = line.strip()
    #print(line)

    toks = line.split("\t")
    print("toks", toks)
    assert(len(toks) == 3)

    score = toks[0]
    url1 = toks[1]
    url2 = toks[2]

    url1Id = GetURLId(mycursor, url1)
    url2Id = GetURLId(mycursor, url2)
    print("   ", url1Id, url2Id)

mydb.commit()

print("Finished")
