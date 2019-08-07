import mysql.connector
import configparser
import time


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
