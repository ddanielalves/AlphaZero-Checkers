from Agents import RandomAgent
from MCTS import MCTS
import pickle
import config
import random
import numpy as  np
from PV_NN import Policy_Value_NN
import Checkers
from Trainer import Trainer 
import dill

c = Checkers.Checkers()

# p1 = RandomAgent()
p2 = Trainer.get_best_agent(c)

# players = {1:p1, -1:p2}
# t = Trainer(c, players = players)
# t.compare_agents()

# p = Policy_Value_NN(c)

# board =np.array([[-1., -0., -1., -0., -1., -0., -1., -0.],
#        [-0., -1., -0., -1., -0., -1., -0., -1.],
#        [-1., -0., -1., -0., -1., -0., -0., -0.],
#        [-0., -0., -0., -0., -0., -0., -0., -1.],
#        [-0., -0., -0., -0., -0., -0., -0., -0.],
#        [-0.,  1., -0.,  1., -0.,  1., -0.,  1.],
#        [ 1., -0.,  1., -0.,  1., -0.,  1., -0.],
#        [-0.,  1., -0.,  1., -0.,  1., -0.,  1.]])
# c.board.pieces = board

# p.predict(c)

# with open("m1.pk", "rb") as pk:
#     data = dill.load(pk)


# t.train(p)


t = Trainer(c)

t.load_players()
t.players[-1] = 
t.load_training_data()

t.train_nn(t.players[1].policy_value_network)
winner = t.compare_agents()

t.update_best_agent(winner)