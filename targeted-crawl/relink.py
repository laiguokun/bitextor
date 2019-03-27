#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################
def Relink(mycursor, docIds, fromId, toId):
    for docId in docIds:
        #print("docId", docId, fromId, toId)

        sql = "UPDATE IGNORE link SET url_id = %s WHERE url_id = %s AND document_id = %s"
        val = (toId, fromId, docId)
        mycursor.execute(sql, val)

    sql = "DELETE FROM link WHERE url_id = %s"
    val = (fromId, )
    mycursor.execute(sql, val)

    sql = "DELETE FROM url WHERE id = %s"
    val = (fromId, )
    mycursor.execute(sql, val)

######################################################################################
def GetDocIds(mycursor):
    sql = "SELECT id FROM document"
    mycursor.execute(sql)
    res = mycursor.fetchall()

    docIds = []
    for rec in res:
        docIds.append(rec[0])

    return docIds

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

    docIds = GetDocIds(mycursor)
    print("docIds", len(docIds))

    sql = "select t1.id, t2.id" \
        + " from url t1, url t2" \
        + " where right(t1.val, 4) = '.htm'" \
        + " and left(t1.val, length(t1.val) - 4) = t2.val" \
        + " and t1.document_id is null" \
        + " and t2.document_id is not null"
        #+ " and t1.id < 1000 and t2.id < 1000"
    mycursor.execute(sql)
    res = mycursor.fetchall()
    print("res", len(res))

    #row = res[0]
    #print("row", row)

    for row in res:
        print("row", row)
        fromId = row[0]
        toId = row[1]
        Relink(mycursor, docIds, fromId, toId)
        mydb.commit()

    mydb.commit()

    print("Finished")

if __name__ == "__main__":
    Main()

