#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector

print("Starting")

oparser = argparse.ArgumentParser(description="import-mysql")
oparser.add_argument('--lang1', dest='l1', help='Language l1 in the crawl', required=True)
oparser.add_argument('--lang2', dest='l2', help='Language l2 in the crawl', required=True)
options = oparser.parse_args()

mydb = mysql.connector.connect(
    host="localhost",
    user="paracrawl_user",
    passwd="paracrawl_password",
    database="paracrawl",
    charset='utf8'
)
mydb.autocommit = False
mycursor = mydb.cursor()

mydb.commit()

print("Finished")
