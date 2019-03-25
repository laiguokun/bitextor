#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################

######################################################################################

print("Starting")

mydb = mysql.connector.connect(
    host="localhost",
    user="paracrawl_user",
    passwd="paracrawl_password",
    database="paracrawl",
    charset='utf8'
)
mydb.autocommit = False
mycursor = mydb.cursor()

sql = "select t1.id, t2.id, t1.val, t2.val" \
    + " from url t1, url t2" \
    + " where right(t1.val, 4) = '.htm'" \
    + " and left(t1.val, length(t1.val) - 4) = t2.val" \
    + " and t1.document_id is null" \
    + " and t2.document_id is not null"



print("Finished")
