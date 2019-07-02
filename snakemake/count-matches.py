#!/usr/bin/env python3

import os
import sys
import argparse
import tldextract
import lzma

print("# Starting")

oparser = argparse.ArgumentParser(description="#docs and sentence matched")
oparser.add_argument('--input-dir', dest='inDir', help='Transient directory', required=True)
oparser.add_argument('--lang', dest='lang', help='Not english language', required=True)
options = oparser.parse_args()

for dir in os.listdir(options.inDir):
  #print("dir", dir)
  dir = "{inDir}/{dir}".format(inDir=options.inDir, dir=dir)

  # doc align
  file = "{dir}/docalign/{lang}-en.customMT.matches".format(dir=dir, lang=options.lang)
  if os.path.isfile(file):
      with open(file) as f:
        lines = f.readlines()
        nunLines = len(lines)
        print(dir, nunLines)
  
print("# Finished")
