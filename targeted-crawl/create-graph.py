#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################
def GetDocId(mycursor, url):
    c = hashlib.md5()
    c.update(url.encode())
    hashURL = c.hexdigest()
    #print("url", url)

    sql = "SELECT id, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    assert(res is not None)

    docId = res[1]
    assert (docId is not None)

    return docId

def ExpandDoc(mycursor, docId):
    sql = "SELECT url_id FROM link WHERE document_id = %s"
    val = (docId,)
    mycursor.execute(sql, val)
    res = mycursor.fetchall()
    print("res", res)

######################################################################################

print("Starting")

oparser = argparse.ArgumentParser(description="create-graph")
oparser.add_argument("--root-page", dest="rootPage", required=True, help="Starting url of domain")
oparser.add_argument('--lang1', dest='l1', help='Language l1 in the crawl', required=True)
oparser.add_argument('--lang2', dest='l2', help='Language l2 in the crawl', required=True)
oparser.add_argument('--out-file', dest='outFile', help='Output dot file', required=True)

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

# root
docId = GetDocId(mycursor, options.rootPage)
print("docId", docId)

ExpandDoc(mycursor, docId)

print("Finished")
