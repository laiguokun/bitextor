#!/usr/bin/env python3

import os
import sys
import argparse
import mysql.connector
import hashlib

######################################################################################
def Relink(mycursor, fromId, toId):
    # delete links in pages which have BOTH a and a.htm
    # would havve been better to use subquery but mysql doesn't support them with same table in the sub
    sql = "select document_id, count(*) as c" \
        + " from link " \
        + " where url_id in (%s, %s)" \
        + " group by document_id" \
        + " having c > 1"
    val = (toId, fromId)
    mycursor.execute(sql, val)
    res = mycursor.fetchall()
    for row in res:
        print("row", row)
        docId = row[0]

        sql = "DELETE FROM link WHERE url_id = %s AND document_id = %s"
        val = (fromId, docId)
        mycursor.execute(sql, val)

    sql = "UPDATE link SET url_id = %s WHERE url_id = %s"
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

    oparser = argparse.ArgumentParser(description="import-mysql")
    oparser.add_argument('--start-id', dest='fromId', help='start id to change', required=True)
    oparser.add_argument('--end-id', dest='toId', help='end id to change', required=True)
    options = oparser.parse_args()

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
        + " and t2.document_id is not null" \
        + " and t1.id BETWEEN %s AND %s"
        #+ " limit 10"
        #+ " and t1.id < 1000 and t2.id < 1000"
    val = (options.fromId, options.toId)
    mycursor.execute(sql, val)
    res = mycursor.fetchall()

    numLines = len(res)
    print("res", numLines)

    #row = res[0]
    #print("row", row)

    lineNum = 0
    for row in res:
        print("row", row, numLines - lineNum)
        fromId = row[0]
        toId = row[1]
        Relink(mycursor, fromId, toId)

        mydb.commit()
        lineNum += 1

    mycursor.close()
    mydb.close()
    mycursor = None
    mydb = None

    print("Finished")

if __name__ == "__main__":
    Main()

