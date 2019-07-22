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
    reliable, text_bytes, detected_languages = cld2.detect(
        data, isPlainText=False)
    return detected_languages[0][1]

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

def SaveURL(mycursor, pageURL, docId, crawlDate):
    origURL = pageURL
    pageURL = NormalizeURL(pageURL)

    c = hashlib.md5()
    c.update(pageURL.encode())
    hashURL = c.hexdigest()
    #print("pageURL", pageURL, hashURL)

    sql = "SELECT id, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is not None:
        # url exists
        urlId = res[0]

        if docId is not None:
            assert(crawlDate is not None)
            if res[1] is None:
                sql = "UPDATE url SET document_id = %s, crawl_date = %s WHERE id = %s"
                val = (docId, crawlDate, urlId)
                mycursor.execute(sql, val)
                assert(mycursor.rowcount == 1)
            else:
                print("WARNING duplicate URL with different document", pageURL, docId, res[1])
                #assert (res[1] == docId)
    else:
        sql = "INSERT INTO url(val, orig_url, md5, document_id, crawl_date) VALUES (%s, %s, %s, %s, %s)"
        # print("url1", pageURL, hashURL)
        val = (pageURL, origURL, hashURL, docId, crawlDate)
        mycursor.execute(sql, val)
        urlId = mycursor.lastrowid

    return urlId

######################################################################################
def SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL, languagesClass):
    if linkStr is not None:
        linkStr = str(linkStr)
        linkStr = linkStr.replace('\n', ' ')

        # translate. Must be 1 sentence
        try:
            linkLangStr = guess_lang_from_data2(linkStr)
            # print("linkLangStr", linkLangStr)
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
        except:
            sys.stderr.write("WARNING: error guessing language")
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
def SaveLinks(mycursor, languages, mtProc, html_text, pageURL, docId, languagesClass):
    #print(html_text)
    soup = BeautifulSoup(html_text, features='html5lib') # lxml html.parser
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
def SaveDoc(mycursor, html_text, pageURL, crawlDate, hashDoc, lang, langId, mime):
    # has URL already been saved, eg. canonical
    normURL = NormalizeURL(pageURL)

    c = hashlib.md5()
    c.update(normURL.encode())
    hashURL = c.hexdigest()

    print("pageURL", pageURL, normURL, hashURL)
    sql = "SELECT id, val, md5, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is not None:
        docId = res[3]
        if docId is not None:
           return (False, docId)

    # has doc already saved
    sql = "SELECT id FROM document WHERE md5 = %s"
    val = (hashDoc,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    #print("SaveDoc", res, hashDoc, pageURL)

    #checking for duplicate content (duplicates are discarded)
    if res is None:
        # new doc
        newDoc = True
        sql = "INSERT INTO document(mime, lang_id, md5) VALUES (%s, %s, %s)"
        val = (mime, langId, hashDoc)
        mycursor.execute(sql, val)
        docId = mycursor.lastrowid
        #print("   SaveDoc new", docId, pageURL)
    else:
        # duplicate page
        newDoc = False
        docId = res[0]
        #print("   SaveDoc duplicate", docId, pageURL)

    urlId = SaveURL(mycursor, pageURL, docId, crawlDate)

    # canonical/alternate links
    soup = BeautifulSoup(html_text, features='html5lib') # lxml html.parser
    for link in soup.findAll('link'):
        url = link.get('href')

        if url is None:
            continue
        url = url.strip()

        canonical = link.get('rel')
        if canonical is None or canonical[0] != "canonical":
            continue

        urlId = SaveURL(mycursor, url, docId, crawlDate)

    return (newDoc, docId)

######################################################################################

def ProcessPage(options, mycursor, languages, mtProc, orig_encoding, text, url, crawlDate, seen_md5, languagesClass):
    print("page", url)
    if url == "unknown":
        logging.info("Unknown page url")
        return

    if orig_encoding == None:
        logging.info("Encoding of document " + url + " could not be identified")

    if len(text) == 0:
        logging.info("Empty page")
        return

    # HTML is then normalized
    logging.info(url + ": cleaning html")
    tree=""
    try:
        cleaner = Cleaner(style=True, links=True, add_nofollow=True, page_structure=False, safe_attrs_only=False)
        cleanhtml = cleaner.clean_html(re.sub('encoding *= *"[^"]+"', '', text, flags=re.IGNORECASE))
        tree = ftfy.fix_text(cleanhtml, fix_entities=False, fix_character_width=False)
        #document = html5lib.parse(fixedtext, treebuilder="lxml", namespaceHTMLElements=False)
        #tree = etree.tostring(document, encoding="utf-8")
    except Exception as ex:
        sys.stderr.write(str(ex)+"\n")
        return
    cleantree = tree.replace("&#160;", " ")
    cleantree = cleantree.replace("\t", " ")

    # lang id
    #printable_str = ''.join(x for x in cleantree if x in string.printable)
    logging.info(url + ": detecting language")
    try:
        lang = guess_lang_from_data2(tree)
        langId = languagesClass.GetLang(lang)
    except:
        sys.stderr.write("error guessing language")
        return
    
    #if len(languages) > 0 and lang not in languages:
    #    logging.info("Language of document " + url + ": " + lang + ". Not among searched languages.")
    #else:
    # If enabled, remove boilerplate HTML
    if options.boilerpipe:
        logging.info(url + ": deboiling html")
        extractor = Extractor(extractor='ArticleExtractor', html=cleantree)
        deboiled = extractor.getHTML()
    else:
        deboiled = cleantree

        # We compute MD5 on the HTML (either normalized one or after boilerpipe if enabled): if we get duplicate
        # files we discard them
        c = hashlib.md5()
        c.update(deboiled.encode())
        hashDoc = c.hexdigest()
        #print("c", hash)
        
        # checking for duplicate content (duplicates are discarded)
        if c.hexdigest() in seen_md5:
            logging.info("Repeated file:\t" + url + "\tfirst occurrence\t" + seen_md5[c.hexdigest()])
            pass
        else:
            # If enabled get text with Alcazar library
            if options.alcazar:
                logging.info(url + ": Getting text with Alcazar")
                btext = alcazar.bodytext.parse_article(deboiled)
                if btext.body_text:
                    plaintext = btext.body_text
                else:
                    plaintext = ""
            # Otherwise use beautifulsoup
            else:
                logging.info(url + ": Getting text with BeautifulSoup")
                soup = BeautifulSoup(deboiled, "lxml")
                for script in soup(["script", "style", "img"]):
                    script.extract()  # rip it out

                plaintext = soup.get_text()
                plaintext = re.sub(r"\n+", "\n",
                                    re.sub(r" *\n *", "\n", re.sub(r" +", " ", re.sub(r"\r", "", plaintext))))

            if len(plaintext) > 0:
                seen_md5[c.hexdigest()] = c.hexdigest()
                # Guessing MIME of the file (checked on original content)
                logging.info(url + ": Getting mime")
                mime = magic.from_buffer(text, mime=True)
                #mimeFile.write(mime.encode() + b"\n")

                #urlFile.write(url.encode() + b"\n")
                #langFile.write(lang.encode() + b"\n")
                #encodingFile.write(orig_encoding.encode() + b"\n")

                b64norm = base64.b64encode(cleantree.encode())
                #normHtmlFile.write(b64norm + b"\n")

                if options.boilerpipe:
                    b64deboil = base64.b64encode(deboiled.encode())
                    #deboilFile.write(b64deboil + b"\n")

                b64text = base64.b64encode(html.unescape(plaintext).encode())



                (newDoc, docId) = SaveDoc(mycursor, text, url, crawlDate, hashDoc, lang, langId, mime)
                #print("docId", docId)

                if newDoc:
                    #urlFile.write(url.encode()+b"\n")
                    #langFile.write(lang.encode()+b"\n")
                    #encodingFile.write(orig_encoding.encode()+b"\n")

                    norm_html = cleantree.encode()

                    # links
                    SaveLinks(mycursor, languages, mtProc, text, url, docId, languagesClass)

                    # write html and text files
                    filePrefix = options.outDir + "/" + str(docId)

                    with lzma.open(filePrefix + ".html.xz", "wt") as htmlFile:
                        htmlFile.write(text)
                    with lzma.open(filePrefix + ".norm.xz", "wt") as normHtmlFile:
                        normHtmlFile.write(norm_html.decode("utf-8"))
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

    seen_md5={}
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
            orig_encoding, text = convert_encoding(payload)
            logging.info("Processing document: " + url)

            if orig_encoding is None:
                logging.info("Encoding of document " + url + " could not be identified")


            ProcessPage(options, mycursor, languages, mtProc, orig_encoding, text, url, crawlDate, seen_md5, languagesClass)

    # everything done
    # commit in case there's any hanging transactions
    mydb.commit()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
