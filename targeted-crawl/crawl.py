#!/usr/bin/env python3
import os
import sys
import requests
import urllib
import argparse
import datetime
import tldextract
import base64
from bs4 import BeautifulSoup

######################################################################################
def strip_scheme(url):
    parsed = urllib.parse.urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, '', 1)

def NormalizeURL(url):
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    if url[-1:] == "/":
        url = url[:-1]

    url = strip_scheme(url)

    return url


######################################################################################
class CrawlHost:
    def __init__(self, url, outDir, maxCount):
        self.url = url
        self.outDir = outDir
        self.maxCount = maxCount
        self.count = 0
        self.visited = set()

        self.domain = tldextract.extract(url).registered_domain
        #print("self.domain", self.domain)

        if os.path.exists(self.outDir):
            if not os.path.isdir(self.outDir):
                sys.stderr.write("Must be a directory: " + self.outDir)
        else:
            os.mkdir(self.outDir)

        self.journal = open(outDir + "/journal", "w")

    def __del__(self):
        print(self.visited)

    ######################################################################################
    def Start(self):
        self.Download("START", self.url, None, None)
    
    ######################################################################################
    def Download(self, parentURL, url, linkStr, imgURL):
        if self.count >= self.maxCount:
            return False

        domain = tldextract.extract(url).registered_domain
        if domain != self.domain:
            return True

        normURL = NormalizeURL(url)
        if normURL in self.visited:
            return True

        self.count += 1
        self.visited.add(normURL)

        pageResponse = requests.get(url, timeout=5)

        for histResponse in pageResponse.history:
            print("   histResponse", histResponse, histResponse.url, histResponse.headers['Content-Type'], \
                    histResponse.apparent_encoding, histResponse.encoding)
            #print(histResponse.text)

            normHistURL = NormalizeURL(histResponse.url)
            self.visited.add(normHistURL)

            self.WriteJournal(parentURL, histResponse.url, histResponse.status_code, linkStr, imgURL)

            parentURL = histResponse.url
            linkStr = None
            imgURL = None
    
        # found page, or error
        print("pageResponse", pageResponse, pageResponse.url, pageResponse.headers['Content-Type'], \
                pageResponse.apparent_encoding, pageResponse.encoding)
        #print(pageResponse.text)

        with open(self.outDir + "/" + str(self.count) + ".text", "w") as f:
            f.write(pageResponse.text)

        with open(self.outDir + "/" + str(self.count) + ".content", "wb") as f:
            f.write(pageResponse.content)

        normPageURL = NormalizeURL(pageResponse.url)
        self.visited.add(normPageURL)

        self.WriteJournal(parentURL, pageResponse.url, pageResponse.status_code, linkStr, imgURL)

        soup = BeautifulSoup(pageResponse.content, features='html5lib') # lxml html.parser
        #soup = BeautifulSoup(pageResponse.text, features='html5lib') # lxml html.parser

        cont = self.FollowLinks(soup, pageResponse.url)
        return cont

    ######################################################################################
    def FollowLinks(self, soup, pageURL):
        coll = soup.findAll('a')

        for link in coll:
            url = link.get('href')
            if url is None:
                continue
            url = url.strip()
            url = urllib.parse.urljoin(pageURL, url)
            
            linkStr = link.string
            print("url", linkStr, url)
            
            imgURL = link.find('img')
            if imgURL:
                #print("imgURL", imgURL, pageURL)
                imgURL = imgURL.get('src')
                if imgURL is not None:
                    imgURL = urllib.parse.urljoin(pageURL, imgURL)
                    imgURL = str(imgURL)
                    #print("   imgURL", imgURL, pageURL)
            else:
                imgURL = None

            cont = self.Download(pageURL, url, linkStr, imgURL)
            if not cont:
                return False

        return True

    ######################################################################################
    def WriteJournal(self, parentURL, url, status_code, linkStr, imgURL):
        if linkStr == None: linkStr = ""
        if imgURL == None: imgURL = ""

        linkStrB64 = base64.b64encode(linkStr.encode()).decode()

        journalStr = str(self.count) +"\t" \
                    + parentURL + "\t" + url + "\t" \
                    + str(status_code) + "\t" \
                    + str(datetime.datetime.now()) + "\t" \
                    + linkStrB64 + "\t" \
                    + imgURL \
                    + "\n"
        self.journal.write(journalStr)

######################################################################################

def Main():
    print("Starting")
    oparser = argparse.ArgumentParser(description="hieu's crawling")
    oparser.add_argument("--url", dest="url", required=True,
                     help="Starting URL to crawl")
    oparser.add_argument("--out-dir", dest="outDir", default=".",
                     help="Directory where html, text and journal will be saved. Will create if doesn't exist")
    oparser.add_argument("--max-requests", dest="maxRequests", default=10000, type=int,
                     help="Max number of user-generated requests")
    options = oparser.parse_args()

    crawler = CrawlHost(options.url, options.outDir, options.maxRequests)
    crawler.Start()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
