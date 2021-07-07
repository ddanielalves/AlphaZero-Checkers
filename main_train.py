import os 
from Checkers import Checkers
from Trainer import Trainer
import cProfile
import pstats
import dill
import pickle


g = Checkers()
t = Trainer(g)

# with cProfile.Profile() as pr:
#     t.train()

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# try:

t.train()
# except Exception as e:
#     print(e)
#     with open("m1.pk", "wb") as pk:
#         del t.players[1].MCTS.policy_value_netwok
#         dill.dump(t.players[1].MCTS, pk)

#     with open("m2.pk", "wb") as pk:
#         del t.players[-1].MCTS.policy_value_netwok
#         dill.dump(t.players[-1].MCTS, pk)