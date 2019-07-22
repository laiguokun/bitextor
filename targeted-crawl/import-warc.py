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
def DocAlign():
    if lang == options.l1:
        otherLang = options.l2
    else:
        otherLang = options.l1

    sql = "SELECT id FROM document WHERE lang=%s"
    val = (otherLang,)
    mycursor.execute(sql, val)
    res = mycursor.fetchall()
    #print("res", res)

    tok1 = "{BITEXTOR}/preprocess/moses/tokenizer/tokenizer.perl -l {lang1} -a -b -q".format(BITEXTOR=BITEXTOR, lang1=options.l1)

    for rec in res:
        otherDocId = rec[0]
        print("other doc id", docId, otherDocId, lang, otherLang)

        if lang == options.l1:
            doc1 = transPath
            doc2 = "{outDir}/{docId}.{lang}.extracted.xz".format(outDir=options.outDir, docId=otherDocId, lang=options.l2)
            matchPath = "{outDir}/{doc1Id}-{doc2Id}.matches".format(outDir=options.outDir, doc1Id=docId, doc2Id=otherDocId)
        else:
            doc1 = "{outDir}/{docId}.trans.xz".format(outDir=options.outDir, docId=otherDocId)
            doc2 = extractPath
            matchPath = "{outDir}/{doc1Id}-{doc2Id}.matches".format(outDir=options.outDir, doc1Id=otherDocId, doc2Id=docId)

        cmd = "{BITEXTOR}/document-aligner/compute_matches.py --lang1 {lang1} --lang2 {lang2} --output_matches {output} --threshold {DOC_THRESHOLD} --word_tokeniser '{WORDTOK1}'".format(BITEXTOR=BITEXTOR, lang1=doc1, lang2=doc2, output=matchPath, DOC_THRESHOLD=0.2, WORDTOK1=tok1)
        #print("cmd", cmd)
        os.system(cmd)

######################################################################################

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

    try:
        url = urllib.parse.urljoin(pageURL, url)
        url = strip_scheme(url)

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
    except:
        sys.stderr.write("error saving link")


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

    # canonical/alternate links
    for link in soup.findAll('link'):
        url = link.get('href')

        if url is None:
            continue
        url = url.strip()

        linkStr = None
        imgURL = None

        SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL, languagesClass)

######################################################################################
def SaveDoc(mycursor, langId, mime):
    sql = "INSERT INTO document(mime, lang_id) VALUES (%s, %s)"
    val = (mime, langId)
    mycursor.execute(sql, val)
    docId = mycursor.lastrowid
    #print("   SaveDoc new", docId, pageURL)

    return docId

######################################################################################
def SaveURL(mycursor, pageURL, docId, crawlDate):
    origURL = pageURL
    pageURL = NormalizeURL(pageURL)

    c = hashlib.md5()
    c.update(pageURL.encode())
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
        val = (pageURL, origURL, hashURL, docId, crawlDate)
        mycursor.execute(sql, val)
        urlId = mycursor.lastrowid

    return urlId

def SaveURLs(mycursor, pageURL, soup, docId, crawlDate):
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

    pageURLId = SaveURL(mycursor, pageURL, docId, crawlDate)
    if canonicalHash is not None:
        SaveURL(mycursor, canonical, docId, crawlDate)

    return pageURLId

######################################################################################
def ProcessPage(options, mycursor, languages, mtProc, orig_encoding, htmlText, url, crawlDate, languagesClass):
    print("page", url)
    if url == "unknown":
        logging.info("Unknown page url")
        return

    if orig_encoding == None:
        logging.info("Encoding of document " + url + " could not be identified")

    if len(htmlText) == 0:
        logging.info("Empty page")
        return

    # lang id
    #printable_str = ''.join(x for x in cleantree if x in string.printable)
    logging.info(url + ": detecting language")
    success, lang = guess_lang_from_data2(htmlText)
    if success:
        langId = languagesClass.GetLang(lang)
    else:
        return
        
    logging.info(url + ": Getting text with BeautifulSoup")
    soup = BeautifulSoup(htmlText, features='html5lib') # lxml html.parser
    for script in soup(["script", "style", "img"]):
        script.extract()  # rip it out

    plaintext = soup.get_text()

    if len(plaintext) > 0:
        # Guessing MIME of the file (checked on original content)
        logging.info(url + ": Getting mime")
        mime = magic.from_buffer(htmlText, mime=True)
        #mimeFile.write(mime.encode() + b"\n")

        docId = SaveDoc(mycursor, langId, mime)
        #print("docId", docId)

        urlId = SaveURLs(mycursor, url, soup, docId, crawlDate)
        #print("docId", docId)

        # links
        SaveLinks(mycursor, languages, mtProc, soup, url, docId, languagesClass)

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

        # doc align
        if 0:
            DocAlign()

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
            url = record.rec_headers.get_header('WARC-Target-URI')[1:-1]
        else:
            url = record.rec_headers.get_header('WARC-Target-URI')
        if url == "unknown":
            logging.info("Skipping page with unknown URL")
            continue
        if "text/dns" in record.rec_headers.get_header('Content-Type'):
            continue
        
        pageSize = int(record.rec_headers.get_header('Content-Length'))
        if pageSize > 5242880:
            logging.info("Skipping page, over limit. " + str(pageSize) + " " + url)
            continue
        if record.http_headers is not None and record.http_headers.get_header('Content-Type') is not None:
            if "image/" in record.http_headers.get_header('Content-Type') or "audio/" in record.http_headers.get_header('Content-Type') or "video/" in record.http_headers.get_header('Content-Type') or "text/x-component" in record.http_headers.get_header('Content-Type') or "text/x-js" in record.http_headers.get_header('Content-Type') or "text/javascript" in record.http_headers.get_header('Content-Type') or "application/x-javascript" in record.http_headers.get_header('Content-Type') or "text/css" in record.http_headers.get_header('Content-Type') or "application/javascript" in record.http_headers.get_header('Content-Type') or "application/x-shockwave-flash" in record.http_headers.get_header('Content-Type') or "application/octet-stream" in record.http_headers.get_header('Content-Type') or "application/x-font-ttf" in record.http_headers.get_header('Content-Type'):
                continue
        url = url.lower()
        if url[-4:] == ".gif" or url[-4:] == ".jpg" or url[-5:] == ".jpeg" or url[-4:] == ".png" or url[-4:] == ".css" or url[-3:] == ".js" or url[-4:] == ".mp3" or url[-4:] == ".mp4" or url[-4:] == ".ogg" or url[-5:] == ".midi" or url[-4:] == ".swf":
            continue
        #print("url", numPages, url, pageSize)

        crawlDate = record.rec_headers.get_header('WARC-Date')
        #print("date", crawlDate)
        crawlDate = crawlDate.replace("T", " ")
        crawlDate = crawlDate.replace("Z", " ")
        crawlDate = crawlDate.strip()
        crawlDate = datetime.strptime(crawlDate, '%Y-%m-%d  %H:%M:%S')
        #print("crawlDate", crawlDate, type(crawlDate))

        payload=record.content_stream().read()
        payloads = []

        if url[-4:] == ".pdf" or ((record.http_headers is not None and record.http_headers.get_header('Content-Type') is not None) and "application/pdf" in record.http_headers.get_header('Content-Type')):
            #if options.pdfextract:
            #    payloads = pdfextract(payload)
            #else:
            #    payloads = pdf2html(payload)
            continue
        elif url[-4:] == ".odt" or url[-4:] == ".ods" or url[-4:] == ".odp":
            #payloads = openoffice2html(payload)
            continue
        elif url[-5:] == ".docx" or url[-5:] == ".pptx" or url[-5:] == ".xlsx":
            #payloads = office2html(payload)
            continue
        elif url[-5:] == ".epub":
            #payloads = epub2html(payload)
            continue
        else:
            payloads = [payload]

        for payload in payloads:
            # We convert into UTF8 first of all
            orig_encoding, htmlText = convert_encoding(payload)
            logging.info("Processing document: " + url)

            if orig_encoding is None:
                logging.info("Encoding of document " + url + " could not be identified")


            ProcessPage(options, mycursor, languages, mtProc, orig_encoding, htmlText, url, crawlDate, languagesClass)

    # everything done
    # commit in case there's any hanging transactions
    mydb.commit()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
