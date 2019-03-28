#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################
def Relink(mycursor, fromId, toId):
    sql = "UPDATE link SET url_id = %s WHERE url_id = %s AND document_id IS NULL"
    val = (toId, fromId)
    mycursor.execute(sql, val)

    sql = "DELETE FROM link WHERE url_id = %s"
    val = (fromId, )
    mycursor.execute(sql, val)

    sql = "DELETE FROM url WHERE id = %s"
    val = (fromId, )
    mycursor.execute(sql, val)

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
    mycursor = mydb.cursor(buffered=True)

    sql = "select t1.id, t2.id" \
        + " from url t1, url t2" \
        + " where right(t1.val, 4) = '.htm'" \
        + " and left(t1.val, length(t1.val) - 4) = t2.val" \
        + " and t1.document_id is null" \
        + " and t2.document_id is not null" #\
        #+ " limit 10"
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
        Relink(mycursor, fromId, toId)
        mydb.commit()

    mycursor.close()
    mydb.close()
    mycursor = None
    mydb = None

    print("Finished")

if __name__ == "__main__":
    Main()

