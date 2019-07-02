#!/usr/bin/env python3

import os
import sys
import argparse
import tldextract
import lzma

#print("# Starting")

oparser = argparse.ArgumentParser(description="#docs and sentence matched")
oparser.add_argument('--input-dir', dest='inDir', help='Transient directory', required=True)
oparser.add_argument('--lang', dest='lang', help='Not english language', required=True)
options = oparser.parse_args()

for domain in os.listdir(options.inDir):
  #print("domain", domain)
  dir = "{inDir}/{domain}".format(inDir=options.inDir, domain=domain)

  # doc align
  file = "{dir}/docalign/{lang}-en.customMT.matches".format(dir=dir, lang=options.lang)
  if os.path.isfile(file):
      with open(file) as f:
        lines = f.readlines()
        nunDocs = len(lines)
        #print(domain, nunDocs)

  # sentence aligned
  file = "{dir}/bleualign.segalign.xz".format(dir=dir)
  if os.path.isfile(file):
    with lzma.open(file) as f:
      lines = f.readlines()
      nunSent = len(lines)
      #print(domain, nunSent)


  # sentence aligned
  file = "{dir}/bleualign.segclean.xz".format(dir=dir)
  if os.path.isfile(file):
    with lzma.open(file) as f:
      lines = f.readlines()
      nunSegcleaned = len(lines)
                                
  # bicleaned sentences
  file = "{dir}/bleualign.bicleaner.xz".format(dir=dir)
  if os.path.isfile(file):
    with lzma.open(file) as f:
      lines = f.readlines()
      nunBicleaned = len(lines)
      #print(domain, nunLines)

  print(domain, nunDocs, nunSent, nunSegcleaned, nunBicleaned, sep='\t')
                                      
  
#print("# Finished")
