#!/usr/bin/env python3
# xzcat www.samsonite.be.xz | ./import-mysql.py  --out-dir out --lang1 en --lang2 fr

#sudo pip3 install mysql-connector-python
import os
import sys
from warcio.archiveiterator import ArchiveIterator
import mysql.connector
import cchardet
import logging
import re
import html5lib
import ftfy
import argparse
import hashlib
import magic
import base64
import html
import lzma
import urllib
import subprocess
import pycld2 as cld2
import string
from lxml.html.clean import Cleaner
from lxml import etree
from bs4 import BeautifulSoup
from datetime import datetime

BITEXTOR = "/home/hieu/workspace/github/paracrawl/bitextor.hieu.targeted"

sys.path.append(BITEXTOR)
from external_processor import ExternalTextProcessor

######################################################################################
class Languages:
    def __init__(self, mycursor):
        self.mycursor = mycursor
        self.coll = {}

    def GetLang(self, str):
        str = StrNone(str)
        if str in self.coll:
            return self.coll[str]
        #print("GetLang", str)

        # new language
        sql = "SELECT id FROM language WHERE lang = %s"
        val = (str,)
        self.mycursor.execute(sql, val)
        res = self.mycursor.fetchone()
        if res is None:
            sql = "INSERT INTO language(lang) VALUES (%s)"
            val = (str,)
            self.mycursor.execute(sql, val)
            langId = self.mycursor.lastrowid
        else:
            langId = res[0]

        #print("langId", langId)
        self.coll[str] = langId

        return langId


def guess_lang_from_data2(data):
    try:
        reliable, text_bytes, detected_languages = cld2.detect(
            data, isPlainText=False)
    except:
        sys.stderr.write("error guessing language")
        return False, None

    return True, detected_languages[0][1]

######################################################################################
def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)
######################################################################################
def convert_encoding(data):
    encoding = cchardet.detect(data)['encoding']
    #print("encoding", data, encoding)

    if encoding is not None and len(data) > 0:
        #We convert, even if the text is detected to be UTF8 so, if it is an error and conversion fails, the error is catched here
        for enc in [encoding, 'utf-8', 'iso-8859-1', 'windows‑1252']:
            try:
                return (enc,data.decode(enc))
            except UnicodeDecodeError:
                sys.stderr.write("encoding error")

    return (None,'')
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
def filter_digits_and_punctuation(original_text):
    text_split = original_text.split()
    if len(text_split) == 1 and sum([1 for m in text_split[0] if m in string.punctuation + string.digits]) > len(
            text_split[0]) // 2:
        return False

    return True

def split_sentences(original_text, sentence_splitter_cmd, prune_type, prune_threshold):
    #print("original_text", len(original_text))
    proc = ExternalTextProcessor(sentence_splitter_cmd.split())

    tmp1 = original_text.replace("\n\n", "\n")
    #print("tmp1", len(tmp1))

    tmp2 = proc.process(tmp1)
    #print("tmp2", len(tmp2))

    tmp3 = html.unescape(tmp2)
    #print("tmp3", len(tmp3))

    tmp4 = [n for n in tmp3.split("\n") if filter_digits_and_punctuation(n)]
    #print("tmp4", len(tmp4))

    tmp5 = []
    count = 0
    for extracted_line in tmp4:
        extracted_line = extracted_line.strip()

        if not extracted_line:
            # print("empty line")
            continue

        if prune_type == "chars":
            if len(extracted_line) > prune_threshold:
                continue
        elif prune_type == "words":
            if len(extracted_line.split()) > prune_threshold:
                continue

        tmp5.append(extracted_line)

        count += 1
    #print("tmp5", len(tmp5))

    return tmp5

######################################################################################
def SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL, languagesClass):
    if linkStr is not None:
        linkStr = str(linkStr)
        linkStr = linkStr.replace('\n', ' ')

        # translate. Must be 1 sentence
        success, linkLangStr = guess_lang_from_data2(linkStr)
        # print("linkLangStr", linkLangStr)
        if success:
            if linkLangStr != languages[-1]:
                tempStr = linkStr + "\n"
                mtProc.stdin.write(tempStr.encode('utf-8'))
                mtProc.stdin.flush()
                linkStrTrans = mtProc.stdout.readline()
                linkStrTrans = linkStrTrans.decode("utf-8")
                linkStrTrans = linkStrTrans.strip("\n")
                # print("linkStr", linkStr, "|||", linkStrTrans)
            else:
                linkStrTrans = linkStr
        else:
            linkStrTrans = None
            linkLangStr = None

    else:
        linkStrTrans = None
        linkLangStr = None

    linkLangId = languagesClass.GetLang(linkLangStr)
    #print("linkLangId", linkLangId)

    url = urllib.parse.unquote(url)
    #print("   URL", pageURL, url)

    url = urllib.parse.urljoin(pageURL, url)

    #print("   link", url, " ||| ", linkStr, " ||| ", imgURL)
    urlId = SaveURL(mycursor, url, None, None)

    sql = "SELECT id FROM link WHERE document_id = %s AND url_id = %s"
    val = (docId, urlId)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is None:
        # not link yet
        if linkStr is None or len(linkStr) < 300: 
            # protect from weird parsing error
            sql = "INSERT INTO link(text, text_lang_id, text_en, hover, image_url, document_id, url_id) VALUES(%s, %s, %s, %s, %s, %s, %s)"
            val = (linkStr, linkLangId, linkStrTrans, "hover here", imgURL, docId, urlId)
            mycursor.execute(sql, val)


######################################################################################
def SaveLinks(mycursor, languages, mtProc, soup, pageURL, docId, languagesClass):
    coll = soup.findAll('a')
    for link in coll:
        url = link.get('href')
        if url is None:
            continue
        url = url.strip()
        
        linkStr = link.string
        #print("url", linkStr, url)
        
        imgURL = link.find('img')
        if imgURL:
            # print("imgURL", imgURL)
            imgURL = imgURL.get('src')
            if imgURL is not None:
                imgURL = str(imgURL)
        else:
            imgURL = None

        SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL, languagesClass)
    #print("coll", len(coll))

######################################################################################
def SaveDoc(mycursor, langId, mime):
    sql = "INSERT INTO document(mime, lang_id) VALUES (%s, %s)"
    val = (mime, langId)
    mycursor.execute(sql, val)
    docId = mycursor.lastrowid
    #print("   SaveDoc new", docId, pageURL)

    return docId

######################################################################################
def SaveURL(mycursor, url, docId, crawlDate):
    normURL = NormalizeURL(url)

    c = hashlib.md5()
    c.update(normURL.encode())
    hashURL = c.hexdigest()
    #print("pageURL", pageURL, hashURL)

    sql = "SELECT id, document_id, crawl_date FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is not None:
        # url exists
        urlId = res[0]

        if docId is not None:
            assert(crawlDate is not None)
            if docId != res[1] or crawlDate != res[2]:
                sql = "UPDATE url SET document_id = %s, crawl_date = %s WHERE id = %s"
                val = (docId, crawlDate, urlId)
                mycursor.execute(sql, val)
                assert(mycursor.rowcount == 1)
    else:
        sql = "INSERT INTO url(val, orig_url, md5, document_id, crawl_date) VALUES (%s, %s, %s, %s, %s)"
        # print("url1", pageURL, hashURL)
        val = (normURL, url, hashURL, docId, crawlDate)
        mycursor.execute(sql, val)
        urlId = mycursor.lastrowid

    return urlId

######################################################################################

def SavePageURLs(mycursor, pageURL, soup, docId, crawlDate):
    c = hashlib.md5()

    c.update(NormalizeURL(pageURL).encode())
    pageURLHash = c.hexdigest()

    canonical = None
    canonicalHash = None
    for link in soup.findAll('link'):
        linkType = link.get('rel')
        if linkType is not None and linkType[0] == "canonical":
            canonical = link.get('href')
            if canonical is not None:
                canonical = canonical.strip()
                c.update(NormalizeURL(canonical).encode())
                canonicalHash = c.hexdigest()
                break

    # does the canonical or page url already have a doc? 
    # If so, use it instead of new doc
    strHashes = pageURLHash
    if canonicalHash is not None:
        strHashes += "," + canonicalHash

    sql = "SELECT id, document_id, crawl_date FROM url WHERE md5 in (%s) AND document_id IS NOT NULL"
    val = (strHashes,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is not None:
        # already has doc, use existing and make url entries consistent
        docId = res[1]
        crawlDate = res[2]

    _ = SaveURL(mycursor, pageURL, docId, crawlDate)
    if canonicalHash is not None:
        SaveURL(mycursor, canonical, docId, crawlDate)

    return docId, canonical

######################################################################################
def ProcessPage(options, mycursor, languages, mtProc, orig_encoding, htmlText, pageURL, crawlDate, languagesClass):
    print("page", pageURL)
    if pageURL == "unknown":
        logging.info("Unknown page url")
        return

    if orig_encoding == None:
        logging.info("Encoding of document " + pageURL + " could not be identified")

    if len(htmlText) == 0:
        logging.info("Empty page")
        return

    # lang id
    #printable_str = ''.join(x for x in cleantree if x in string.printable)
    logging.info(pageURL + ": detecting language")
    success, lang = guess_lang_from_data2(htmlText)
    if success:
        langId = languagesClass.GetLang(lang)
    else:
        return
        
    logging.info(pageURL + ": Getting text with BeautifulSoup")
    soup = BeautifulSoup(htmlText, features='html5lib') # lxml html.parser
    for script in soup(["script", "style", "img"]):
        script.extract()  # rip it out

    plaintext = soup.get_text()

    if len(plaintext) > 0:
        # Guessing MIME of the file (checked on original content)
        logging.info(pageURL + ": Getting mime")
        mime = magic.from_buffer(htmlText, mime=True)
        #mimeFile.write(mime.encode() + b"\n")

        docId = SaveDoc(mycursor, langId, mime)
        #print("docId", docId)

        docId, canonical = SavePageURLs(mycursor, pageURL, soup, docId, crawlDate)
        #print("docId", docId)

        # canonical links
        if canonical is not None:
            SaveLink(mycursor, languages, mtProc, pageURL, docId, canonical, None, None, languagesClass)

        # links
        SaveLinks(mycursor, languages, mtProc, soup, pageURL, docId, languagesClass)

        # write html and text files
        filePrefix = options.outDir + "/" + str(docId)

        with lzma.open(filePrefix + ".html.xz", "wt") as htmlFile:
            htmlFile.write(htmlText)
        with lzma.open(filePrefix + ".text.xz", "wt") as textFile:
            textFile.write(plaintext)

        #print("plaintext", len(plaintext))
        splitterCmd = "{BITEXTOR}/preprocess/moses/ems/support/split-sentences.perl -b -l {lang1}".format(BITEXTOR=BITEXTOR, lang1=lang)
        extractedLines = split_sentences(plaintext, splitterCmd, options.prune_type, options.prune_threshold)

        # write splitted file
        extractPath = options.outDir + "/" + str(docId) + "." + lang + ".extracted.xz"
        with lzma.open(extractPath, 'wt') as extractFile:
            for extractedLine in extractedLines:
                extractFile.write(str(docId) + "\t" + extractedLine + "\n")

        if lang != languages[-1]:
            # translate
            transPath = options.outDir + "/" + str(docId) + ".trans.xz"
            transFile = lzma.open(transPath, 'wt')

            for inLine in extractedLines:
                # print("inLine", inLine)
                inLine += "\n"
                mtProc.stdin.write(inLine.encode('utf-8'))
                mtProc.stdin.flush()
                outLine = mtProc.stdout.readline()
                outLine = outLine.decode("utf-8")
                transFile.write(str(docId) + "\t" + outLine)

            transFile.close()

######################################################################################
def Main():
    print("Starting")

    oparser = argparse.ArgumentParser(description="import-mysql")
    oparser.add_argument("--boilerpipe", action="store_true", default=False, help="Use boilerpipe bodytext to do the de-boiling")
    oparser.add_argument("--alcazar", action="store_true", default=False, help="Use alcazar bodytext extract relevant text from HTML. By default BeautifulSoup4is used")
    oparser.add_argument('--langs', dest='langs', help='Languages in the crawl. Last is the dest language', required=True)
    oparser.add_argument('--out-dir', dest='outDir', help='Output directory', required=True)
    oparser.add_argument("--prune", dest="prune_threshold", type=int,
                        default=80, help="Prune sentences longer than n (words/characters)", required=False)
    oparser.add_argument("--prune_type", dest="prune_type", choices={"words", "chars"},
                        default="words", help="Prune sentences either by words or charaters", required=False)
    oparser.add_argument("--verbose", action="store_true", default=False,
                         help="Produce additional information about preprocessing through stderr.")
    options = oparser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO if options.verbose else logging.ERROR, datefmt='%Y-%m-%d %H:%M:%S')

    languages = options.langs.split(",")
    assert(len(languages) == 2)

    mydb = mysql.connector.connect(
        host="localhost",
        user="paracrawl_user",
        passwd="paracrawl_password",
        database="paracrawl",
        charset='utf8'
    )
    mydb.autocommit = False
    mycursor = mydb.cursor()

    f = ArchiveIterator(sys.stdin.buffer)
    languagesClass = Languages(mycursor)

    magic.Magic(mime=True)

    mtProc = subprocess.Popen(["/home/hieu/workspace/experiment/issues/paracrawl/phi-system/translate-pipe.sh",
                             languages[0]
                             ],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    numPages = 0

    for record in f:
        numPages += 1
        if numPages % 1 == 0:
            pass
            #print("write", numPages)
            mydb.commit()

        if record.rec_type != 'response':
            continue
        if record.rec_headers.get_header('WARC-Target-URI')[0] == '<' and record.rec_headers.get_header('WARC-Target-URI')[-1] == '>':
            pageURL = record.rec_headers.get_header('WARC-Target-URI')[1:-1]
        else:
            pageURL = record.rec_headers.get_header('WARC-Target-URI')
        if pageURL == "unknown":
            logging.info("Skipping page with unknown URL")
            continue
        if "text/dns" in record.rec_headers.get_header('Content-Type'):
            continue
        
        pageSize = int(record.rec_headers.get_header('Content-Length'))
        if pageSize > 5242880:
            logging.info("Skipping page, over limit. " + str(pageSize) + " " + pageURL)
            continue
        if record.http_headers is not None and record.http_headers.get_header('Content-Type') is not None:
            if "image/" in record.http_headers.get_header('Content-Type') or "audio/" in record.http_headers.get_header('Content-Type') or "video/" in record.http_headers.get_header('Content-Type') or "text/x-component" in record.http_headers.get_header('Content-Type') or "text/x-js" in record.http_headers.get_header('Content-Type') or "text/javascript" in record.http_headers.get_header('Content-Type') or "application/x-javascript" in record.http_headers.get_header('Content-Type') or "text/css" in record.http_headers.get_header('Content-Type') or "application/javascript" in record.http_headers.get_header('Content-Type') or "application/x-shockwave-flash" in record.http_headers.get_header('Content-Type') or "application/octet-stream" in record.http_headers.get_header('Content-Type') or "application/x-font-ttf" in record.http_headers.get_header('Content-Type'):
                continue
        pageURL = pageURL.lower()
        if pageURL[-4:] == ".gif" or pageURL[-4:] == ".jpg" or pageURL[-5:] == ".jpeg" or pageURL[-4:] == ".png" or pageURL[-4:] == ".css" or pageURL[-3:] == ".js" or pageURL[-4:] == ".mp3" or pageURL[-4:] == ".mp4" or pageURL[-4:] == ".ogg" or pageURL[-5:] == ".midi" or pageURL[-4:] == ".swf":
            continue
        #print("pageURL", numPages, pageURL, pageSize)

        crawlDate = record.rec_headers.get_header('WARC-Date')
        #print("date", crawlDate)
        crawlDate = crawlDate.replace("T", " ")
        crawlDate = crawlDate.replace("Z", " ")
        crawlDate = crawlDate.strip()
        crawlDate = datetime.strptime(crawlDate, '%Y-%m-%d  %H:%M:%S')
        #print("crawlDate", crawlDate, type(crawlDate))

        payload=record.content_stream().read()
        payloads = []

        if pageURL[-4:] == ".pdf" or ((record.http_headers is not None and record.http_headers.get_header('Content-Type') is not None) and "application/pdf" in record.http_headers.get_header('Content-Type')):
            #if options.pdfextract:
            #    payloads = pdfextract(payload)
            #else:
            #    payloads = pdf2html(payload)
            continue
        elif pageURL[-4:] == ".odt" or pageURL[-4:] == ".ods" or pageURL[-4:] == ".odp":
            #payloads = openoffice2html(payload)
            continue
        elif pageURL[-5:] == ".docx" or pageURL[-5:] == ".pptx" or pageURL[-5:] == ".xlsx":
            #payloads = office2html(payload)
            continue
        elif pageURL[-5:] == ".epub":
            #payloads = epub2html(payload)
            continue
        else:
            payloads = [payload]

        assert(len(payloads) == 1)
        # We convert into UTF8 first of all
        #print(payload)
        orig_encoding, htmlText = convert_encoding(payloads[0])
        logging.info("Processing document: " + pageURL)

        if orig_encoding is None:
            logging.info("Encoding of document " + pageURL + " could not be identified")


        ProcessPage(options, mycursor, languages, mtProc, orig_encoding, htmlText, pageURL, crawlDate, languagesClass)

    # everything done
    # commit in case there's any hanging transactions
    mydb.commit()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
