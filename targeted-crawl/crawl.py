#!/usr/bin/env python3
import os
import sys
import requests
import urllib
from bs4 import BeautifulSoup

######################################################################################
#helpers
def ConvertEncoding(data, encoding):
    if encoding is not None and len(data) > 0:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            sys.stderr.write("encoding error")
    return ''

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
    def __init__(self, url, maxCount):
        self.url = url
        self.maxCount = maxCount
        self.count = 0
        self.visited = set()
        self.journal = open("journal", "w")

    def __del__(self):
        print(self.visited)

    ######################################################################################
    def Start(self):
        self.Download("START", self.url)
    
    ######################################################################################
    def Download(self, parentURL, url):
        if self.count >= self.maxCount:
            return False

        normURL = NormalizeURL(url)
        if normURL in self.visited:
            return True

        self.count += 1
        self.visited.add(normURL)

        pageResponse = requests.get(url, timeout=5)
        print("status_code", pageResponse.status_code)

        for histResponse in pageResponse.history:
            print("   histResponse", histResponse, histResponse.url, histResponse.headers['Content-Type'], \
                    histResponse.apparent_encoding, histResponse.encoding)
            #print(histResponse.text)
            self.WriteJournal(parentURL, histResponse.url, histResponse.status_code)

            histURL =  histResponse.url
            normHistURL = NormalizeURL(histURL)
            self.visited.add(normHistURL)

            parentURL = histResponse.url
    

        print("pageResponse", pageResponse, pageResponse.url, pageResponse.headers['Content-Type'], \
                pageResponse.apparent_encoding, pageResponse.encoding)
        #print(pageResponse.text)

        dirName = str(self.count)
        os.mkdir(dirName)
        with open(dirName + "/text", "w") as f:
            f.write(pageResponse.text)

        with open(dirName + "/content", "wb") as f:
            f.write(pageResponse.content)

        self.WriteJournal(parentURL, pageResponse.url, pageResponse.status_code)

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
                # print("imgURL", imgURL)
                imgURL = imgURL.get('src')
                if imgURL is not None:
                    imgURL = str(imgURL)
            else:
                imgURL = None

            cont = self.Download(pageURL, url)
            if not cont:
                return False

        return True

    ######################################################################################
    def WriteJournal(self, parentURL, url, status_code):
        journalStr = str(self.count) +"\t" + parentURL + "\t" + url + "\t" + str(status_code) + "\n"
        self.journal.write(journalStr)

######################################################################################

def Main():
    print("Starting")

    #url = "http://www.visitbritain.com"
    url = "http://www.buchmann.ch"
    #url = "https://www.buchmann.ch/catalog/default.php"
    crawler = CrawlHost(url, 7)
    crawler.Start()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
