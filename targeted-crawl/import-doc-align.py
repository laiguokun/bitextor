#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib
import urllib
import html5lib

######################################################################################
def strip_scheme(url):
    parsed = urllib.parse.urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, '', 1)

def NormalizeURL(url):
    url = url.lower()
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    if url[-1:] == "/":
        url = url[:-1]
    if url[-5:] == ".html":
        url = url[:-5] + ".htm"
        #print("pageURL", pageURL)
    if url[-9:] == "index.htm":
        url = url[:-9]

    url = strip_scheme(url)

    return url
######################################################################################

def GetDocId(mycursor, url):
    c = hashlib.md5()
    c.update(url.encode())
    hashURL = c.hexdigest()
    #print("url", url, hashURL)

    sql = "SELECT t1.id, t2.id FROM url t1, response t2 " \
        + "WHERE t2.url_id = t1.id " \
        + "AND t1.md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    
    assert(res is not None)
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
    url1 = toks[1]
    url2 = toks[2]

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
