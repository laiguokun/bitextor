#!/usr/bin/env python3
import os
import sys
import requests
from bs4 import BeautifulSoup

######################################################################################
def ConvertEncoding(data, encoding):
    if encoding is not None and len(data) > 0:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            sys.stderr.write("encoding error")
    return ''

######################################################################################
class CrawlHost:
    def __init__(self, url, maxCount):
        self.url = url
        self.maxCount = 100
        self.count = 0

    ######################################################################################
    def Start(self):
        self.Download(self.url)
    
    ######################################################################################
    def Download(self, url):
        self.count += 1
        if self.count > self.maxCount:
            return False

        pageResponse = requests.get(url, timeout=5)
        print("status_code", pageResponse.status_code)

        for histResponse in pageResponse.history:
            print("   histResponse", histResponse, histResponse.url, histResponse.headers['Content-Type'], \
                    histResponse.apparent_encoding, histResponse.encoding)
            #print(histResponse.text)

        print("pageResponse", pageResponse, pageResponse.url, pageResponse.headers['Content-Type'], \
                pageResponse.apparent_encoding, pageResponse.encoding)
        #print(pageResponse.text)

        text = pageResponse.text
        #text = ConvertEncoding(pageResponse.text, pageResponse.encoding)

        content = pageResponse.content

        dirName = str(self.count)
        os.mkdir(dirName)
        with open(dirName + "/text", "w") as f:
            f.write(text)

        with open(dirName + "/content", "wb") as f:
            f.write(content)

        soup = BeautifulSoup(content, features='html5lib') # lxml html.parser
        #soup = BeautifulSoup(pageResponse.text, features='html5lib') # lxml html.parser

        self.FollowLinks(soup)

    ######################################################################################
    def FollowLinks(self, soup):
        coll = soup.findAll('a')

        for link in coll:
            url = link.get('href')
            if url is None:
                continue
            url = url.strip()
            
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


######################################################################################

def Main():
    print("Starting")

    url = "http://www.visitbritain.com"
    #url = "http://www.buchmann.ch"
    #url = "https://www.buchmann.ch/catalog/default.php"
    crawler = CrawlHost(url, 7)
    crawler.Start()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
