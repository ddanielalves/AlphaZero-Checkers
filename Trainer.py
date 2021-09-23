""" Environment used in self play to train the Neural Networks 

Includes:
- Trainer: Object that manages the self play and stores the training data
"""

from collections import deque
# import logging
import pickle
import numpy as np
import os.path
from tqdm import tqdm
import random 

import Agents
from PV_NN import Policy_Value_NN
import config


class Trainer:
    def __init__(self, game, load_nn = False, players = None) -> None:

        self.game = game
        
        self.replay_memory = deque(maxlen=config.MAX_QUEUE)

        self.EPISODES = config.EPISODES
        self.TRAIN_EVERY = config.TRAIN_EVERY

        if players is None:
            self.load_players()
        else:
            self.players = players

    def load_players(self):
        """Load the players from previous training iterations
        """
        player1 = Trainer.get_best_agent(self.game)
        player2 = Trainer.get_best_agent(self.game)

        self.players = {1: player1, -1: player2}

    @staticmethod
    def get_best_agent(game):
        """Get the best player stored as the best

        Args:
            game (Game): Game in which to load the Neural Network

        Returns:
            Agent: The stochastic agent used for training
        """

        net = Policy_Value_NN(game, load_best=True)
        return Agents.StochasticAgent(net)

    def update_replay_memory(self, training_data):
        """Update the memory used for training de NN

        Args:
            training_data (Tuple): Game iterations that are going to be added to the training data
        """

        self.replay_memory += training_data

    def train(self, load_train = True):
        """Train the agent and check if there is a new best agent

        Args:
            load_train (bool, optional): Flag to ask if it is neccessary to load the stored training data. Defaults to True.
        """

        if load_train:
            self.load_training_data()

        print("Starting to train...")
        for i in tqdm(range(config.EPISODES)):
            self.play_training_game()

            self.save_training_data()

            if i != 0 and i % config.TRAIN_EVERY == 0 and len(self.replay_memory) >= config.BATCH_SIZE:
                self.train_nn(self.players[1].policy_value_network)
                winner_agent = self.compare_agents()
                self.update_best_agent(winner_agent)
                self.load_players()
            
        


    def compare_agents(self):
        """Play COMPARING_GAMES games between the previous best agent and the recently trained agent
        to check if there is a new best agent.

        Returns:
            Agent: The current best agent
        """

        winners = {-1:0, 0:0, 1: 0}

        print(f"Playing {config.COMPARING_GAMES} game between the agents")
        for i in tqdm(range(config.COMPARING_GAMES)):
            winner = self.play_training_game()
            
            winners[winner] += 1
        
        winner_agent = 1 if winners[1] >= winners[-1] else -1
        print(winners)
        return self.players[winner_agent]


    def update_best_agent(self, agent):
        """Save the current best agent

        Args:
            agent (Agent): Current best agent
        """

        agent.policy_value_network.save_model(f"./trained_models/best_{self.game.nrows}.h5")


    def train_nn(self, nn):
        """Train the neural network with a random sample of the replay memory

        Args:
            nn (Policy_Value_NN): NN object that is going to be trained
        """

        if len(self.replay_memory) >= config.BATCH_SIZE:
            batch = random.sample(self.replay_memory, config.BATCH_SIZE)
            
            X = np.array([data["board"] for data in batch])
            X = X.reshape(X.shape[0], 1, self.game.nrows, self.game.nrows)

            y_value = np.array([data["value"] for data in batch])

            y_policy = np.zeros([len(batch), self.game.ACTION_SPACE_SIZE])
            for i, data in enumerate(batch):
                for action in data["action_prob"]:
                    y_policy[i, action] = data["action_prob"][action]        

            nn.train(X, y_policy, y_value)

    def reset(self):
        """Reset the players and the game for a new one to be played
        """

        self.players[1].reset()
        self.players[-1].reset()
        self.game.reset()


    def play_training_game(self):
        """Play a training game. 
        This method is used both during the training and evaluation of the agents.

        Returns:
            winner: Winner of the game or 0 if it's a draw
        """

        self.reset()
        done = False
        training_data = []
        player_turn = self.game.players_turn

        while not done:
            player = self.players[player_turn]
            action, action_prob = player.predict(self.game)
            self.move_players_mcts_root(action)
            iteration = {"player":player_turn, "board":self.game.state.copy(), "action": action, "action_prob":action_prob}
            # print('state', iteration["board"])
            training_data.append(iteration)
            # break
            done, winner, player_turn = self.game.step(action)
            # print('state2', iteration["board"],"-" * 20, training_data[-1])

            if done:
                self.backpropagete_value(training_data, winner)
        
        self.update_replay_memory(training_data) 
        return winner
        
    def save_training_data(self):
        """Save the training data as pickle object
        """

        with open(f"training_data/replay_memory_{self.game.nrows}.pickle", "wb") as pick:
            pickle.dump(self.replay_memory, pick)
     
    
    def load_training_data(self):
        """Load the training data for the current board size
        """

        if os.path.isfile(f"training_data/replay_memory_{self.game.nrows}.pickle"):
            with open(f"training_data/replay_memory_{self.game.nrows}.pickle", "rb") as pick:
                self.replay_memory = pickle.load(pick)


    def move_players_mcts_root(self, action):
        """Move the MCTS root after a move is played

        Args:
            action (action): Action took. It's used to move the root of the MCTS
        """

        if self.players[1].has_MCTS:
            self.players[1].MCTS.move_root(action)
        if self.players[-1].has_MCTS:
            self.players[-1].MCTS.move_root(action)


    def backpropagete_value(self, training_data, winner):
        """After a game is finished backpropagate the winner to the previous nodes

        Args:
            training_data (list): list containing the training data of the current game to update the winner
            winner (int): Winner of the game played

        Returns:
            training_data: training data updated with the winner
        """
        
        for i in range(len(training_data)):
            if winner == training_data[i]["player"]:
                value = 1 
            elif winner == 0:
                value = 0
            else:
                value = -1
            training_data[i]["value"] = value
        return training_data



