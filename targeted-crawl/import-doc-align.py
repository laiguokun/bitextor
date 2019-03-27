#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib


######################################################################################
def NormalizeURL(url):
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    if url[-5:].lower() == ".html":
        pageURL = url[:-5] + ".htm"
        #print("pageURL", pageURL)
    return url

def GetDocId(mycursor, url):
    c = hashlib.md5()
    c.update(url.encode())
    hashURL = c.hexdigest()

    sql = "SELECT id, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    assert(res is not None)

    docId = res[1]
    assert (docId is not None)

    return docId

######################################################################################
def SaveDocAlign(mycursor, doc1Id, doc2Id, score):
    sql = "INSERT INTO document_align(document1, document2, score) VALUES (%s, %s, %s)"
    val = (doc1Id, doc2Id, score)
    mycursor.execute(sql, val)

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
    url1 = NormalizeURL(toks[1])
    url2 = NormalizeURL(toks[2])

    doc1Id = GetDocId(mycursor, url1)
    doc2Id = GetDocId(mycursor, url2)
    print("   ", doc1Id, doc2Id)

    SaveDocAlign(mycursor, doc1Id, doc2Id, score)

mydb.commit()

print("Finished")
