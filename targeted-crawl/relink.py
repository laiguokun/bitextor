#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################
def Relink(mycursor, fromId, toId):
    pass
    #sql =

######################################################################################

def Main():
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

    sql = "select t1.id, t2.id" \
        + " from url t1, url t2" \
        + " where right(t1.val, 4) = '.htm'" \
        + " and left(t1.val, length(t1.val) - 4) = t2.val" \
        + " and t1.document_id is null" \
        + " and t2.document_id is not null"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    print("res", len(res))

    row = res[0]
    print("row", row)

    #for row in res:
    #    fromId = row[0]
    #    toId = row[1]
    #    Relink(fromId, toId)

    print("Finished")

if __name__ == "__main__":
    Main()

