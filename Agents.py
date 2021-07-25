import numpy as np

from MCTS import MCTS
from MCTS_Original import MCTS as M_O
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


class OriginalMCTSAgent:
    def __init__(self):
        self.has_MCTS = True 
        self.reset()

    def reset(self):
        self.MCTS = M_O()


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
        # print(self.MCTS.root.children)
        print(self.MCTS.root.children.keys())

        for child_action in self.MCTS.root.children:

            action_n = self.MCTS.root.children[child_action].N
            action_ns.append(action_n)

            if  action_n > max_n:
                max_n = action_n
                action = child_action
        return action, dict(zip(list(self.MCTS.root.children.keys()), action_ns))




class StochasticAgent:
    def __init__(self, policy_value_network):
        self.has_MCTS = True 
        self.policy_value_network = policy_value_network
        self.reset()

    def reset(self):
        self.MCTS = MCTS(self.policy_value_network)


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



class DetermenisticAgent:
    def __init__(self, policy_value_network):
        self.has_MCTS = True
        self.policy_value_network = policy_value_network
        self.reset()

    def reset(self):
        self.MCTS = MCTS(self.policy_value_network)

        
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

        
