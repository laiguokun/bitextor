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

def FollowLinks(soup):
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


url = "http://www.visitbritain.com"
#url = "http://www.buchmann.ch"
#url = "https://www.buchmann.ch/catalog/default.php"
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

with open("text", "w") as f:
    f.write(text)

with open("content", "wb") as f:
    f.write(content)

soup = BeautifulSoup(content, features='html5lib') # lxml html.parser
#soup = BeautifulSoup(pageResponse.text, features='html5lib') # lxml html.parser

FollowLinks(soup)