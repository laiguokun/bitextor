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

def GetURL(mycursor, url):
    c = hashlib.md5()
    c.update(url.lower().encode())
    hashURL = c.hexdigest()
    #print("url", url, hashURL)

    sql = "SELECT id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    urlId = None
    if res is not None:
        # url exists
        urlId = res[0]

    return urlId

######################################################################################
def SaveURLAlign(mycursor, urlId1, urlId2, score):
    print("SaveURLAlign", urlId1, urlId2, score)
    sql = "INSERT INTO url_align(url1, url2, score) VALUES (%s, %s, %s)"
    val = (urlId1, urlId2, score)

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

    urlId1 = GetURL(mycursor, url1)
    if urlId1 is None:
        raise Exception("URL not found:" + urlId1)

    urlId2 = GetURL(mycursor, url2)
    if urlId2 is None:
        raise Exception("URL not found:" + urlId2)

    #print("   ", urlId1, urlId2, toks)

    SaveURLAlign(mycursor, urlId1, urlId2, score)

mydb.commit()

print("Finished")
