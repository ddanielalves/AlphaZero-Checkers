import os 
from Checkers import Checkers
from Trainer import Trainer
import cProfile
import pstats
import dill
import pickle


g = Checkers(max_moves = 200, nrows=8)
t = Trainer(g)


t.train()


