#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib


######################################################################################
def NormalizeURL(url):
    url = url.lower()
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    if url[-5:] == ".html":
        url = url[:-5] + ".htm"
        #print("pageURL", pageURL)
    if url[-9:] == "index.htm":
        url = url[:-9]

    if url[:7] == "http://":
        #print("   strip protocol1", url, url[7:])
        url = url[7:]
    elif url[:8] == "https://":
        #print("   strip protocol2", url, url[8:])
        url = url[8:]

    return url

def GetDocId(mycursor, url):
    c = hashlib.md5()
    c.update(url.encode())
    hashURL = c.hexdigest()
    #print("url", url, hashURL)

    sql = "SELECT id, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    
    if res is not None:
        docId = res[1]
        if docId is not None:
            return docId

    print("WARNING: doc not found for URL", url)
    return None

######################################################################################
def SaveDocAlign(mycursor, doc1Id, doc2Id, score):
    print("SaveDocAlign", doc1Id, doc2Id, score)
    sql = "INSERT INTO document_align(document1, document2, score) VALUES (%s, %s, %s)"
    val = (doc1Id, doc2Id, score)

    try:
        # duplicatess, possibly due to URL normalization
        mycursor.execute(sql, val)
    except:
        sys.stderr.write("encoding error")

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
    #print("toks", toks)
    assert(len(toks) == 3)

    score = toks[0]
    url1 = NormalizeURL(toks[1])
    url2 = NormalizeURL(toks[2])

    doc1Id = GetDocId(mycursor, url1)
    if doc1Id is None:
        continue

    doc2Id = GetDocId(mycursor, url2)
    if doc2Id is None:
        continue

    print("   ", doc1Id, doc2Id, toks)

    SaveDocAlign(mycursor, doc1Id, doc2Id, score)

mydb.commit()

print("Finished")
