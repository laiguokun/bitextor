#!/usr/bin/env python3

import os
import sys
import argparse

print("Starting")

oparser = argparse.ArgumentParser(description="create-graph")
oparser.add_argument("--root-page", dest="rootPage", required=True, help="Starting url of domain")



print("Finished")
