import os
import sys
import numpy as np

######################################################################################
class Corpus:
    def __init__(self, params):
        self.params = params
        self.transitions = []
        self.losses = []

    def AddTransition(self, transition):    
        self.transitions.append(transition)
