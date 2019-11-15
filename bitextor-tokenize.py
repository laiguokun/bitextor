#!/usr/bin/env python3

#  This file is part of Bitextor.
#
#  Bitextor is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Bitextor is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Bitextor.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import argparse
import base64
import lzma

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/utils")
from utils.common import open_xz_or_gzip_or_plain
from utils.tokenization import filter_digits_and_punctuation
from utils.tokenization import SentenceSplitter
from utils.tokenization import Tokenizer


def extract_encoded_text(encoded, sent_proc, tok_proc, morph_proc):
    content = base64.b64decode(encoded).decode("utf-8").replace("\t", " ")
    segments = sent_proc.split_sentences(content, "")

    segments_filtered = [n for n in segments.split("\n") if filter_digits_and_punctuation(n)]

    tokenized_text = []
    for segment in segments_filtered:
        tokenized_text.append(tok_proc.tokenize(segment))

    if morph_proc:
        lemmatized_text = []
        for segment in tokenized_text:
            lemmatized_text.append(morph_proc.tokenize(segment))
        tokenized_text = lemmatized_text

    b64sentences = base64.b64encode(("\n".join(segments_filtered)+"\n").encode("utf-8"))
    b64tokenized = base64.b64encode(("\n".join(tokenized_text)+"\n").lower().encode("utf-8"))
    return b64sentences, b64tokenized


oparser = argparse.ArgumentParser(
    description="Tool that tokenizes (sentences, tokens and morphemes) plain text")
oparser.add_argument('--text', dest='text', help='Plain text file', required=True)
oparser.add_argument('--sentence-splitter', dest='splitter', required=True, help="Sentence splitter commands")
oparser.add_argument('--word-tokenizer', dest='tokenizer', required=True, help="Word tokenisation command")
oparser.add_argument('--morph-analyser', dest='lemmatizer', help="Morphological analyser command")
oparser.add_argument('--sentences-output', default="plain_sentences.xz", dest='sent_output', help="Path of the output file that will contain sentence splitted text")
oparser.add_argument('--tokenized-output', default="plain_tokenized.xz", dest='tok_output', help="Path of the output file that will contain sentence splitted and tokenized text")

options = oparser.parse_args()

with open_xz_or_gzip_or_plain(options.text) as reader, lzma.open(options.sent_output, "w") as sent_writer, lzma.open(options.tok_output, "w") as tok_writer:
    sent_proc = SentenceSplitter(options.splitter.split())
    tok_proc = Tokenizer(options.tokenizer.split())
    if options.lemmatizer:
        morph_proc = Tokenizer(options.lemmatizer.split())
    else:
        morph_proc = None
    for line in reader:
        encoded_text = line.strip()
        sentences, tokenized = extract_encoded_text(encoded_text, sent_proc, tok_proc, morph_proc)
        if sentences and tokenized:
            sent_writer.write(sentences + b"\n")
            tok_writer.write(tokenized + b"\n")
