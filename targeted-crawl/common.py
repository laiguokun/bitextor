import mysql.connector
import configparser
import time

def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)

class Timer:
    def __init__(self):
        self.starts = {}
        self.cumm = {}

    def __del__(self):
        print("Timers:")
        for key, val in self.cumm.items():
            print(key, "\t", val)

    def Start(self, str):
        self.starts[str] = time.time()

    def Pause(self, str):
        now = time.time()
        then = self.starts[str]

        if str in self.cumm:
            self.cumm[str] += now - then
        else:
            self.cumm[str] = now - then


class MySQL:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.mydb = mysql.connector.connect(
        host=config["mysql"]["host"],
        user=config["mysql"]["user"],
        passwd=config["mysql"]["password"],
        database=config["mysql"]["database"],
        charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)

class Languages:
    def __init__(self, mycursor):
        self.mycursor = mycursor
        self.coll = {}

        sql = "SELECT id, lang FROM language"


        self.mycursor.execute(sql)
        ress = self.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            self.coll[res[1]] = res[0]
            self.maxLangId = res[0]
        
    def GetLang(self, str):
        str = StrNone(str)
        if str in self.coll:
            return self.coll[str]
        # print("GetLang", str)

        # new language
        sql = "SELECT id FROM language WHERE lang = %s"
        val = (str,)
        self.mycursor.execute(sql, val)
        res = self.mycursor.fetchone()
        assert(res is not None)
        langId = res[0]

        # print("langId", langId)
        self.coll[str] = langId

        return langId
