"""Agents used to play the games

Includes:
- RandomAgent: Agent that takes actions at random, with uniform probability
- OriginalMCTSAgent: Agent that uses the original MCTS to select the actions
- StochasticAgent: Agent that follows the MCTS as it is described in the AlphaZero's paper. It selects its actions in a stochastic way
- DetermenisticAgent: Agent that follows the MCTS as it is described in the AlphaZero's paper. It selects the action that maximizes the final reward 
- HumanAgent: Agent that asks for user input to select the action to take
"""


import numpy as np

from MCTS import MCTS
from MCTS_Original import MCTS as M_O
from Checkers import Board
import config

class RandomAgent:
    """Random Agent
    When it recieces a game object it returns a random valid action
    """

    def __init__(self):
        self.has_MCTS = False

    def predict(self, game):
        valid_moves = game.get_valid_moves()
        i = np.random.choice(len(valid_moves))

        action = game.board.move_to_action(*valid_moves[i])
        return action, []


class OriginalMCTSAgent:
    """ Deterministic agent for the original MCTS
    The main difference between this and the deterministic agent is in the simulation before selecting an action.
    When a leaf node is found, instead of querying the nn, this agent simulates the game until the end, 
    selecting random actions for both players.
    """

    def __init__(self):
        self.has_MCTS = True 
        self.reset()

    def reset(self):
        """Reset the Monte Carlo Tree so a new game can begin
        """
        self.MCTS = M_O()


    def predict(self, game):
        """Predict the action to be taken

        Args:
            game (Game): Game with the board in the state to be predicted

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """

        valid_actions = game.get_valid_actions()
        
        if len(valid_actions) == 1:
            return valid_actions[0],  {valid_actions[0]: 1}

        self.MCTS.run_simulations(game)
        action, action_probs = self.get_deterministic_action()
        return action, action_probs

    def get_deterministic_action(self):
        """After the simulations are runned, compute the action to take based on the most visited node

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """

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
    """Agent that follows the MCTS as it is described in the AlphaZero's paper. It selects its actions in a stochastic way
    """
    
    def __init__(self, policy_value_network):
        self.has_MCTS = True 
        self.policy_value_network = policy_value_network
        self.reset()

    def reset(self):
        """Reset the Monte Carlo Tree, so a new game can be played
        """
        self.MCTS = MCTS(self.policy_value_network)


    def predict(self, game):
        """Predict the action to be taken

        Args:
            game (Game): Game with the board in the state to be predicted

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """

        valid_actions = game.get_valid_actions()
        
        if len(valid_actions) == 1:
            return valid_actions[0],  {valid_actions[0]: 1}

        self.MCTS.run_simulations(game)
        action, action_probs = self.get_stochastic_action()
        return action, action_probs


    def get_stochastic_action(self):
        """Select the action to take given the action probabilities.
        In this function the action to take is selected stochasticaly.

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """

        action_probs = self.get_action_probabilities()
        
        return np.random.choice(list(action_probs.keys()), p = list(action_probs.values())), action_probs


    def get_action_probabilities(self):
        """Calculate the probability of selecting each valid action.
        The probabilities are a function of the number of visits in each node.

        Returns:
            action_probs: dict of probabilities of selecting each action
        """

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



class DetermenisticAgent:
    """Agent that follows the MCTS as it is described in the AlphaZero's paper. 
    It selects the action that maximizes the final reward
    """

    def __init__(self, policy_value_network):
        self.has_MCTS = True
        self.policy_value_network = policy_value_network
        self.reset()

    def reset(self):
        """Reset the Monte Carlo Tree, so a new game can be played
        """

        self.MCTS = MCTS(self.policy_value_network)

        
    def predict(self, game):
        """Predict the action to be taken

        Args:
            game (Game): Game with the board in the state to be predicted

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """

        valid_actions = game.get_valid_actions()
        
        if len(valid_actions) == 1:
            return valid_actions[0],  {valid_actions[0]: 1}

        self.MCTS.run_simulations(game)
        action, action_probs = self.get_deterministic_action()
        return action, action_probs

    def get_deterministic_action(self):
        """Select the action to take given the action probabilities.
        In this function the action that takes to the most visited node is selected.

        Returns:
            action: Action to take
            action_probs: dict of probabilities for each action
        """


        max_n = 0
        action = None
        action_ns = []

        for child_action in self.MCTS.root.children:

            action_n = self.MCTS.root.children[child_action].N
            action_ns.append(action_n)

            if  action_n > max_n:
                max_n = action_n
                action = child_action
        return action, dict(zip(list(self.MCTS.root.children.keys()), action_ns))



class HumanAgent:
    """Agent that asks for user input to select the action to take
    """
    def __init__(self):
        self.has_MCTS = False


    def predict(self, game):
        valid_moves = game.get_valid_moves()
        print(f"valid moves: {valid_moves}")

        piece = int(input("select a piece: "))
        direction = int(input("select a direction: "))
        return Board.move_to_action(piece, direction), None

        
