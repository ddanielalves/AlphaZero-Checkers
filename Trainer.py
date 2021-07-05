from collections import deque
import logging
import pickle
from unicodedata import ucd_3_2_0 
import numpy as np
import os.path
from tqdm import tqdm
import random 

import Agents
from PV_NN import Policy_Value_NN
import config

class Trainer:
    def __init__(self, game, load_nn = False, players = None) -> None:
        # TODO: Implement Loading of NN and checking if exists

        self.game = game
        
        self.replay_memory = deque(maxlen=config.MAX_QUEUE)

        self.EPISODES = config.EPISODES
        self.TRAIN_EVERY = config.TRAIN_EVERY

        if players is None:
            self.load_players()
        else:
            self.players = players

    def load_players(self):
        player1 = Trainer.get_best_agent(self.game)
        player2 = Trainer.get_best_agent(self.game)

        self.players = {1: player1, -1: player2}

    @staticmethod
    def get_best_agent(game):
        net = Policy_Value_NN(game, config.BEST_NN_PATH)
        return Agents.StochasticAgent(net)

    def update_replay_memory(self, training_data):
        self.replay_memory += training_data

    def train(self, load_train = True):
        if load_train:
            self.load_training_data()

        print("Starting to train...")
        for i in tqdm(range(config.EPISODES)):
            logging.info(f"Starting Iteration {i+1}")
            self.play_training_game()
            self.save_training_data()

            if i != 0 and i % config.TRAIN_EVERY == 0 and len(self.replay_memory) >= config.BATCH_SIZE:
                self.train_nn(self.players[1].policy_value_network)
                winner_agent = self.compare_agents()
                self.update_best_agent(winner_agent)
                self.load_players()


    def compare_agents(self):
        winners = {-1:0, 0:0, 1: 0}

        print(f"Playing {config.COMPARING_GAMES} between the agents")
        for i in tqdm(range(config.COMPARING_GAMES)):
            winner = self.play_training_game()
            
            winners[winner] += 1
        
        winner_agent = 1 if winners[1] >= winners[-1] else -1
        print(winners)
        return self.players[winner_agent]

    def update_best_agent(self, agent):
        agent.policy_value_network.save_model(config.BEST_NN_PATH)


    def train_nn(self, nn):
        # TODO: Implement
        batch = random.sample(self.replay_memory, config.BATCH_SIZE)
        
        X = np.array([data["board"] for data in batch])
        X = X.reshape(X.shape[0], 1, 8, 8)

        y_value = np.array([data["value"] for data in batch])

        y_policy = np.zeros([len(batch), self.game.ACTION_SPACE_SIZE])
        for i, data in enumerate(batch):
            for action in data["action_prob"]:
                y_policy[i, action] = data["action_prob"][action]        

        nn.train(X, y_policy, y_value)

    def reset(self):
        if self.players[1].has_MCTS:
            self.players[1].MCTS.reset()
        if self.players[-1].has_MCTS:
            self.players[-1].MCTS.reset()
        self.game.reset()


    def play_training_game(self):
        self.reset()
        done = False
        training_data = []
        player_turn = self.game.players_turn

        while not done:
            player = self.players[player_turn]
            action, action_prob = player.predict(self.game)
            self.move_players_mcts_root(action)

            iteration = {"player":player_turn, "board":self.game.state, "action": action, "action_prob":action_prob}
            training_data.append(iteration)

            done, winner, player_turn = self.game.step(action)
            if done:
                self.backpropagete_value(training_data, winner)
        
        self.update_replay_memory(training_data) 
        return winner
        
    def save_training_data(self):
        with open("training_data/replay_memory.pickle", "wb") as pick:
            pickle.dump(self.replay_memory, pick)
     
    
    def load_training_data(self):
        if os.path.isfile("training_data/replay_memory.pickle"):
            with open("training_data/replay_memory.pickle", "rb") as pick:
                self.replay_memory = pickle.load(pick)


    def move_players_mcts_root(self, action):
        if self.players[1].has_MCTS:
            self.players[1].MCTS.move_root(action)
        if self.players[-1].has_MCTS:
            self.players[-1].MCTS.move_root(action)


    def backpropagete_value(self, training_data, winner):
        for i in range(len(training_data)):
            if winner == training_data[i]["player"]:
                value = 1 
            elif winner == 0:
                value = 0
            else:
                value = -1
            training_data[i]["value"] = value
        return training_data



