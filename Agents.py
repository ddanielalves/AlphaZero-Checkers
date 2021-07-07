import numpy as np

from MCTS import MCTS
import config
from Checkers import Board

class RandomAgent:
    def __init__(self):
        self.has_MCTS = False

    def predict(self, game):
        valid_moves = game.get_valid_moves()
        i = np.random.choice(len(valid_moves))
        # print(f"RANDOM AGENT: move selected: {valid_moves[i]}")
        action = game.board.move_to_action(*valid_moves[i])
        return action, []


class StochasticAgent:
    def __init__(self, policy_value_network):
        self.has_MCTS = True 
        self.policy_value_network = policy_value_network
        self.MCTS = MCTS(policy_value_network)

    def predict(self, game):
        valid_actions = game.get_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0], {valid_actions[0]: 1}

        self.MCTS.run_simulations(game)

        action_prob = self.get_action_probabilities()
        action = self.get_stochastic_action(action_prob)

        return action, action_prob

    def get_action_probabilities(self):
        children_actions = self.MCTS.get_root_children()
        sum_n = 0
        for action in children_actions:
            sum_n += children_actions[action].N

        action_prob = {}
        for action in children_actions:
            node = children_actions[action]
            
            u = (node.N ** (1/config.TAU)) / (sum_n ** (1/config.TAU))
            action_prob[action] = u
        return action_prob


    def get_stochastic_action(self, action_prob= None):
        if action_prob is None:
            action_prob = self.get_action_probabilities()
        
        return np.random.choice(list(action_prob.keys()), p= list(action_prob.values()))




class DetermenisticAgent:
    def __init__(self, policy_value_network):
        self.has_MCTS = True
        self.policy_value_network = policy_value_network
        self.MCTS = MCTS(policy_value_network)


    def predict(self, game):
        valid_actions = game.get_valid_actions()
        
        if len(valid_actions) == 1:
            return valid_actions[0],  {valid_actions[0]: 1}

        self.MCTS.run_simulations(game)
        action, action_probs = self.get_deterministic_action()
        return action, action_probs

    def get_deterministic_action(self):
        max_n = 0
        action = None
        total_n = 0
        action_ns = []
        for child_action in self.MCTS.root.children:

            action_n = self.MCTS.root.children[child_action].N
            action_ns.append(action_n)

            if  action_n > max_n:
                max_n = action_n
                action = child_action
        return action, dict(zip(list(self.MCTS.root.children.keys()), action_ns))



class HumanAgent:
    def __init__(self):
        self.has_MCTS = False


    def predict(self, game):
        valid_moves = game.get_valid_moves()
        print(f"valid moves: {valid_moves}")

        piece = int(input("select a piece: "))
        direction = int(input("select a direction: "))
        return Board.move_to_action(piece, direction), None

        
