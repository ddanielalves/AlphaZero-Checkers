from os import stat
from types import MethodType
from Agents import OriginalMCTSAgent as Agent
from MCTS_Original import MCTS
import pickle
import config
import random
import numpy as  np
from PV_NN import Policy_Value_NN
from Checkers import Checkers as Game
from Trainer import Trainer 
import dill
import sys 
import tracemalloc

import cProfile
import pstats



# t = Trainer(c)

# t.load_players()
# t.load_training_data()

            
    # t.save_training_data()

# t.train_nn(t.players[1].policy_value_network)


import time
def main():
    game = Game(pieces=None)

    m = MCTS()

    t1 = time.time()

    for i in range(100):
        m.run_one_simulation(game)
        # print(np.array(game.board.pieces))

with cProfile.Profile() as pr:
    main()

stats=pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats(filename="profile.prof")
stats.print_stats(100)

# game = Game(pieces=None)
# print(game.board.pieces)
# print(game.nr_to_coord(21))



# agent = Agent(net)

# action, prior = agent.predict(game)

# with open("m.pik", "wb") as p:
#     dill.dump(agent.MCTS, p)
