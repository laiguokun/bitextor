#!/usr/bin/env python3


try:
    import lzma
except ImportError:
    from backports import lzma
import argparse
import base64
import gzip
import os
import string
import sys
import html

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../utils")
from external_processor import ExternalTextProcessor
from common import open_xz_or_gzip_or_plain

def filter_digits_and_punctuation(original_text):
    text_split = original_text.split()
    if len(text_split) == 1 and sum([1 for m in text_split[0] if m in string.punctuation + string.digits]) > len(
            text_split[0]) // 2:
        return False

    return True


def split_sentences(original_text, sentence_splitter_cmd):
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

    return tmp4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", dest="languages",
                        help="Languages to be extracted (comma-separated)", required=True)
    parser.add_argument("--splitter", dest="splitter",
                        help="Sentence splitting script", required=True)
    parser.add_argument("--output_prefix", dest="output_prefix", default="",
                        help="Prefix for output files within directory", required=False)
    parser.add_argument("--output_dir", dest="output_dir", default=".",
                        help="Output directory. Extracted files will be named as <lang>.extracted.gz", required=False)
    parser.add_argument("--prune", dest="prune_threshold", type=int,
                        default=80, help="Prune sentences longer than n (words/characters)", required=False)
    parser.add_argument("--prune_type", dest="prune_type", choices={"words", "chars"},
                        default="words", help="Prune sentences either by words or charaters", required=False)
    parser.add_argument("-x", "--xz", dest="xz", action="store_true",
                        help="Use xz as the compression tool")
    parser.add_argument('--langfile', dest='langFile', help='File containing the language code of each HTML file')
    parser.add_argument('--plaintextfile', dest='textFile',
                        help='File containing the plain text extracted from the HTML documents in a WARC file, '
                             'encoded in base64')
    parser.add_argument('--urlfile', dest='urlFile',
                        help='File containing the list of urls of the documents in a WARC file')

    args = parser.parse_args()

    langs_parse = args.languages.strip().split(',')
    # print("langs_parse", langs_parse)

    lang_file = {}
    for l in langs_parse:
        if not l.strip():
            continue
        if args.xz:
            lang_file[l] = lzma.open(os.path.join(
                args.output_dir, "{0}{1}.extracted.xz".format(args.output_prefix, l)), "wb")
        else:
            lang_file[l] = gzip.open(os.path.join(
                args.output_dir, "{0}{1}.extracted.gz".format(args.output_prefix, l)), "wb")

    with open_xz_or_gzip_or_plain(args.textFile) as text_reader, \
            open_xz_or_gzip_or_plain(args.langFile) as lang_reader, \
            open_xz_or_gzip_or_plain(args.urlFile) as url_reader:
        for line in text_reader:
            text = base64.b64decode(line.strip()).decode("utf-8")
            lang = next(lang_reader, None).strip()
            uri = next(url_reader, None).strip()

            if lang not in langs_parse:
                continue

            if not text:
                continue

            count = 0
            for extracted_line in split_sentences(text, args.splitter):
                extracted_line = extracted_line.strip()
                #print("extracted_line", len(extracted_line))
                if not extracted_line:
                    #print("empty line")
                    continue

                # prune long sentences
                extracted_line = extracted_line
                if args.prune_type == "chars":
                    if len(extracted_line) > args.prune_threshold:
                        continue
                elif args.prune_type == "words":
                    if len(extracted_line.split()) > args.prune_threshold:
                        continue

                lang_file[lang].write("{0}\t{1}\n".format(
                    uri, extracted_line).encode("utf-8"))

                count += 1
            #print("count", count)

        # print("lang_file", lang_file)
        for f in lang_file:
            lang_file[f].close()
