from warnings import resetwarnings
import numpy as np
from tensorflow.python.module.module import valid_identifier
import gc
import config


class Node:
    def __init__(self, parent) -> None:
        """Create a node in the Monte Carlo Tree

        Args:
            parent (Node): Parent Node, None if the current Node is the root
            prior_probability (float): probability of selecting the action that leads to the current Node
        """

        self.parent = parent
        self.N = 0
        self.W = 0
        self.Q = 0

        self.children = {}

    def __repr__(self):
        # return f"Node {self.N=} {self.W=} {self.Q=:.2f} {self.P=:.1f}"
        return f"Node {self.N=} {self.Q=:.2f}"

    def expand(self, valid_actions):
        """On a leaf node, expand will create the children Node 

        Args:
            prior_probabilities (np.array): Array of probabilities of selecting each action given a current state.  
        """

        for action in valid_actions:
            if action not in self.children:
                self.children[action] = Node(self)

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
        sum_n = self.N
        action_values = []
        actions = list(children_actions.keys())
        for action in actions:
            node = children_actions[action]
            u = config.C_PUCT * np.sqrt(np.log(1+sum_n)/(1+node.N))
            action_values.append(node.Q + u)
        return actions[np.argmax(action_values)]

    

class MCTS:
    def __init__(self) -> None:
        """Create a Monte Carto Tree Seach object

        Args:
            policy_value_netwok (Policy_Value_NN): Policy Value Network object
        """

        self.reset()
        
    
    def reset(self):
        """Resets the root node
        """
        self.root = Node(None)

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


    def backup_values(self, leaf, winner, turns):
        """After a value is found by querying the value network of by finishing a game, propagate through the Tree
        The value is updated as -value in the nodes played by the oposite player that took the current action.

        Args:
            leaf (Node): Leaf Node explored
            value (float): Value to be updated throughout the Tree
            player (int): Player that took the last action.
        """

        node = leaf
        while node is not self.root:
            if node.player == winner:
                value_aux = config.REWARD_WINNING/(turns+1)
            else:
                value_aux = -config.REWARD_WINNING/(turns+1)
        
            node.update_value(value_aux)
            node = node.parent
        node.update_value(0)


    def run_simulations(self, game):
        """Run NUM_SIMULATIONS to update the current Tree, before selecting a move.

        Args:
            game (Game): Game with the current state
        """

        for _ in range(config.NUM_SIMULATIONS):
            self.run_one_simulation(game)


    def simulate(self, game):
        done, winner = game.game_finished()
        turns = 0
        while not done:
            action = np.random.choice(game.get_valid_actions())
            done, winner, _, reward = game.step(action)
            turns += 1
        return winner, turns



    def run_one_simulation(self, old_game):
        # print('finding leaf', game.board.pieces)
        # print(old_game.players_turn)
        leaf, game, turns_ = self.find_leaf(old_game)
        leaf.expand(game.get_valid_actions())
        
        winner, turns = self.simulate(game)
        # print(self.root.children, winner, turns+turns_)
        # aux={i:self.root.children[i].N for i in self.root.children}
        # print(game.players_pieces,aux)

        # If the game has ended backup the ending value.
        # Instead of the one that came from the value network.
        self.backup_values(leaf, winner, turns+turns_)
        


    def find_leaf(self, game_):
        game = game_.copy()
        node = self.root
        player = game.get_player_turn()
        turns = 0
        while not node.is_leaf():
            action = node.get_simulation_action()
            # print(game.get_valid_actions(), action)
            _, _, _, reward = game.step(action)

            node = node.children[action]
            node.player = player
            player = game.get_player_turn()
            turns+=1
            # if reward != 0:
            #     # print(game.board.pieces, reward, node.player, action)
            #     self.backup_values(node, reward, node.player)
        node.board = game.board.pieces
        return node, game, turns