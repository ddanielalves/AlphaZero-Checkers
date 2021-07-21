from warnings import resetwarnings
import numpy as np
from tensorflow.python.module.module import valid_identifier
import gc
import config


class Node:
    def __init__(self, parent, prior_probability) -> None:
        """Create a node in the Monte Carlo Tree

        Args:
            parent (Node): Parent Node, None if the current Node is the root
            prior_probability (float): probability of selecting the action that leads to the current Node
        """

        self.parent = parent
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior_probability

        self.children = {}

    def expand(self, prior_probabilities):
        """On a leaf node, expand will create the children Node 

        Args:
            prior_probabilities (np.array): Array of probabilities of selecting each action given a current state.  
        """

        for action in prior_probabilities:
            if action not in self.children:
                self.children[action] = Node(self, prior_probabilities[action])

    def is_leaf(self):
        """Check if the current node is a leaf

        Returns:
            bool: True if the current node is a leaf
        """
        return len(self.children) == 0
    
    def update_value(self, value):
        """Update the current values of a Node.
        The visits count increments by one.
        The Weight increments by the value provided
        Q value updates as the average weight per visit

        Args:
            value (float): Value given by the value Network of by the end of a game
        """
        self.N += 1
        self.W += value
        self.Q = self.W/self.N


    def get_simulation_action(self):
        """Get the action to take on the current Node.
        The action selected is the one that maximizes the following function
            f = Q + u,
        where Q is the Q value of each child node.
        
        And u is given by:
        u = C_PUCT * node.P * np.sqrt(sum_n)/(1+node.N), 
        where sum_n is the sum of the visit counts of every children.


        Returns:
            int: action to take
        """
        children_actions = self.children
        sum_n = 0
        for action in children_actions:
            sum_n += children_actions[action].N

        action_values = []
        actions = list(children_actions.keys())
        for action in actions:
            node = children_actions[action]
            
            u = config.C_PUCT * node.P * np.sqrt(sum_n)/(1+node.N)
            action_values.append(node.Q + u)
 
        return actions[np.argmax(action_values)]

    

class MCTS:
    def __init__(self, policy_value_netwok) -> None:
        """Create a Monte Carto Tree Seach object

        Args:
            policy_value_netwok (Policy_Value_NN): Policy Value Network object
        """

        self.policy_value_netwok = policy_value_netwok
        self.reset()
        
    
    def reset(self):
        """Resets the root node
        """
        self.root = Node(None,0)

    def move_root(self, action):
        """Move the root node to follow the action selected.
        The root must be moved after each action taken by any player/

        Args:
            action (int): Action selected by one of the players
        """
        if self.root.children != {}: 
            new_root = self.root.children[action]
            del new_root.parent
            
            self.root = new_root


    def get_root_children(self):
        """Get the children of the current root node

        Returns:
            dict: Children of the current root node
        """
        return self.root.children


    def backup_values(self, leaf, value, player):
        """After a value is found by querying the value network of by finishing a game, propagate through the Tree
        The value is updated as -value in the nodes played by the oposite player that took the current action.

        Args:
            leaf (Node): Leaf Node explored
            value (float): Value to be updated throughout the Tree
            player (int): Player that took the last action.
        """
        node = leaf
        while node is not self.root:
            if node.player == player:
                value_aux = value
            else:
                value_aux = -value

            node.update_value(value_aux)
            node = node.parent
    
    def run_simulations(self, game):
        """Run NUM_SIMULATIONS to update the current Tree, before selecting a move.

        Args:
            game (Game): Game with the current state
        """
        for _ in range(config.NUM_RAND_SIMLUATIONS):
            self.run_one_simulation(game, random=True)

        for _ in range(config.NUM_SIMULATIONS):
            self.run_one_simulation(game)


    def run_one_simulation(self, game, random=False):
        # print('finding leaf', game.board.pieces)
        leaf, game = self.find_leaf(game)
        
        end, winner = game.game_finished()        
        player = game.get_player_turn()

        # aux={i:self.root.children[i].N for i in self.root.children}
        # print(game.players_pieces,aux)

        # If the game has ended backup the ending value.
        # Instead of the one that came from the value network.
        if end:
            # print("There is a winner", winner)
            if winner == player:
                value = 1
            elif winner == 0:
                value = 0
            else:
                value = -1
        else:
            if random:
                value = 0
                valid_actions = game.get_valid_actions()
                policy = dict(zip(valid_actions, [1/len(valid_actions)]*len(valid_actions)))
            else:
                policy, value = self.policy_value_netwok.predict(game)
            leaf.expand(policy)

        self.backup_values(leaf, value, player)
        


    def find_leaf(self, game_):
        game = game_.copy()
        node = self.root
        node.player = game.get_player_turn()
        while not node.is_leaf():
            action = node.get_simulation_action()
            # print(game.get_valid_actions(), action)
            _, _, _, reward = game.step(action)

            node = node.children[action]
            node.player = game.get_player_turn()

            if reward != 0:
                # print(game.board.pieces, reward, node.player, action)
                self.backup_values(node, reward, node.player)
        node.board = game.board.pieces
        return node, game