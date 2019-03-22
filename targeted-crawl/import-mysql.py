#!/usr/bin/env python3
# xzcat www.samsonite.be.xz | ./import-mysql.py  --out-dir out --lang1 en --lang2 fr

#sudo pip3 install mysql-connector-python
import os
import sys
import warc
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


sys.path.append("..")
from external_processor import ExternalTextProcessor

######################################################################################
def guess_lang_from_data2(data):
    reliable, text_bytes, detected_languages = cld2.detect(
        data, isPlainText=False)
    return detected_languages[0][1]

######################################################################################
def convert_encoding(data):
    encoding = cchardet.detect(data)['encoding']

    if len(data) > 0:
        #We convert, even if the text is detected to be UTF8 so, if it is an error and conversion fails, the error is catched here
        for enc in [encoding, 'utf-8', 'iso-8859-1', 'windows‑1252']:
            try:
                return (enc,data.decode(enc))
            except UnicodeDecodeError:
                pass

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

    tok1 = "../preprocess/moses/tokenizer/tokenizer.perl -l {lang1} -a -b -q".format(lang1=options.l1)

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

        cmd = "/home/hieu/workspace/github/paracrawl/bitextor.hieu.targeted/document-aligner/compute_matches.py --lang1 {lang1} --lang2 {lang2} --output_matches {output} --threshold {DOC_THRESHOLD} --word_tokeniser '{WORDTOK1}'".format(lang1=doc1, lang2=doc2, output=matchPath, DOC_THRESHOLD=0.2, WORDTOK1=tok1)
        #print("cmd", cmd)
        os.system(cmd)

######################################################################################
def SaveURL(mycursor, pageURL, docId):
    ind = pageURL.find("#")
    if ind >= 0:
        pageURL = pageURL[:ind]
        #print("pageURL", pageURL)
    if pageURL[-5:].lower() == ".html":
        pageURL = pageURL[:-5] + ".htm"
        #print("pageURL", pageURL)

    c = hashlib.md5()
    c.update(pageURL.encode())
    hashURL = c.hexdigest()

    sql = "SELECT id, document_id FROM url WHERE md5 = %s"
    val = (hashURL,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()

    if res is not None:
        # url exists
        urlId = res[0]

        if docId is not None:
            if res[1] is None:
                sql = "UPDATE url SET document_id = %s WHERE md5 = %s"
                val = (docId, hashURL)
                mycursor.execute(sql, val)
            else:
                assert (res[1] == docId)
    else:
        sql = "INSERT INTO url(val, md5, document_id) VALUES (%s, %s, %s)"
        # print("url1", pageURL, hashURL)
        val = (pageURL, hashURL, docId)
        mycursor.execute(sql, val)
        urlId = mycursor.lastrowid

    return urlId

######################################################################################
def SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL):
    if linkStr is not None:
        linkStr = str(linkStr)
        linkStr = linkStr.replace('\n', ' ')

        # translate. Must be 1 sentence
        langLinkStr = guess_lang_from_data2(linkStr)
        # print("langLinkStr", langLinkStr)
        if langLinkStr != languages[-1]:
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
        langLinkStr = None

    url = urllib.parse.unquote(url)
    url = urllib.parse.urljoin(pageURL, url)
    url = strip_scheme(url)

    # print("link", url, " ||| ", linkStr, " ||| ", imgURL)
    urlId = SaveURL(mycursor, url, None)

    sql = "INSERT INTO link(text, text_lang, text_en, hover, image_url, document_id, url_id) VALUES(%s, %s, %s, %s, %s, %s, %s)"

    val = (linkStr, langLinkStr, linkStrTrans, "hover here", imgURL, docId, urlId)
    mycursor.execute(sql, val)


######################################################################################
def SaveLinks(mycursor, languages, mtProc, html_text, pageURL, docId):
    soup = BeautifulSoup(html_text, features="lxml")
    for link in soup.findAll('a'):
        url = link.get('href')

        if url is None:
            continue

        linkStr = link.string

        imgURL = link.find('img')
        if imgURL:
            # print("imgURL", imgURL)
            imgURL = imgURL.get('src')
            if imgURL is not None:
                imgURL = str(imgURL)
        else:
            imgURL = None

        SaveLink(mycursor, languages, mtProc, pageURL, docId, url, linkStr, imgURL)

######################################################################################

def ProcessPage(options, mycursor, languages, mtProc, orig_encoding, html_text, pageURL):
    print("page", pageURL)

    if pageURL == "unknown":
        logging.info("Unknown page url")
        return

    if orig_encoding == None:
        logging.info("Encoding of document " + pageURL + " could not be identified")

    if len(html_text) == 0:
        logging.info("Empty page")
        return

    # HTML is then normalized
    cleaner = Cleaner(style=True, links=True, add_nofollow=True, page_structure=False, safe_attrs_only=False)

    cleanhtml = cleaner.clean_html(re.sub('encoding *= *"[^"]+"', '', html_text, flags=re.IGNORECASE))
    document = html5lib.parse(ftfy.fix_text(cleanhtml, fix_entities=False, fix_character_width=False), treebuilder="lxml", namespaceHTMLElements=False)
    tree = etree.tostring(document, encoding="utf-8")
    cleantree = tree.decode("utf8").replace("&#160;", " ")
    cleantree = cleantree.replace("\t", " ")

    # lang id
    lang = guess_lang_from_data2(cleantree)

    #If enabled, remove boilerplate HTML
    if options.boilerpipe:
        extractor = Extractor(extractor='ArticleExtractor', html=cleantree)
        deboiled = extractor.getHTML()
    else:
        deboiled = cleantree

    #We compute MD5 on the HTML (either normalized one or after boilerpipe if enabled): if we get duplicate files we discard them
    c = hashlib.md5()
    c.update(deboiled.encode())
    hashDoc = c.hexdigest()
    #print("c", hash)

    sql = "SELECT id FROM document WHERE md5 = %s"
    val = (hashDoc,)
    mycursor.execute(sql, val)
    res = mycursor.fetchone()
    #print("page", res, hashDoc, pageURL)

    #checking for duplicate content (duplicates are discarded)
    if res is not None:
        # duplicate page
        docId = res[0]

        SaveURL(mycursor, pageURL, docId)
        return

    # new doc
    if options.alcazar:
        # get text with Alcazar library
        btext = alcazar.bodytext.parse_article(cleantree)
        if btext.body_text:
            plaintext = btext.body_text
        else:
            plaintext = ""
    else:
        # use beautifulsoup
        if options.boilerpipe:
            soup = BeautifulSoup(deboiled, "lxml")
        else:
            soup = BeautifulSoup(cleantree, "lxml")
        for script in soup(["script", "style", "img"]):
            script.extract()    # rip it out

        plaintext = soup.get_text()
        plaintext = re.sub(r"\n+","\n",re.sub(r" *\n *","\n",re.sub(r" +"," ",re.sub(r"\r","", plaintext))))

    if len(plaintext) == 0:
        # empty doc. Should we still go thru links anyway?
        return

    #Guessing MIME of the file (checked on original content)
    mime=magic.from_buffer(html_text, mime=True)
    #mimeFile.write(mime.encode()+b"\n")

    #urlFile.write(url.encode()+b"\n")
    #langFile.write(lang.encode()+b"\n")
    #encodingFile.write(orig_encoding.encode()+b"\n")

    norm_html = cleantree.encode()

    sql = "INSERT INTO document(mime, lang, md5) VALUES (%s, %s, %s)"
    val = (mime, lang, hashDoc)
    #print("val", type(val))
    mycursor.execute(sql, val)
    docId = mycursor.lastrowid

    SaveURL(mycursor, pageURL, docId)

    # links
    SaveLinks(mycursor, languages, mtProc, html_text, pageURL, docId)

    # write html and text files
    filePrefix = options.outDir + "/" + str(docId)

    with lzma.open(filePrefix + ".html.xz", "wt") as htmlFile:
        htmlFile.write(html_text)
    with lzma.open(filePrefix + ".norm.xz", "wt") as normHtmlFile:
        normHtmlFile.write(norm_html.decode("utf-8"))
    with lzma.open(filePrefix + ".text.xz", "wt") as textFile:
        textFile.write(plaintext)

    #print("plaintext", len(plaintext))
    splitterCmd = "../preprocess/moses/ems/support/split-sentences.perl -b -l {lang1}".format(lang1=lang)
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
    options = oparser.parse_args()

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

    f = warc.WARCFile(fileobj=sys.stdin.buffer)
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

        #We convert into UTF8 first of all
        orig_encoding,html_text = convert_encoding(record.payload.read())
        pageURL=record.url

        ProcessPage(options, mycursor, languages, mtProc, orig_encoding, html_text, pageURL)

    # everything done
    # commit in case there's any hanging transactions
    mydb.commit()

    print("Finished")

######################################################################################

if __name__ == "__main__":
    Main()
