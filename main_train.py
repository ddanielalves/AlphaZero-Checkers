import os 
from Checkers import Checkers
from Trainer import Trainer


g = Checkers(max_moves = 200, nrows=8)
t = Trainer(g)

t.train()