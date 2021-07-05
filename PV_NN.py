import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Activation, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

import config

class Policy_Value_NN:
    def __init__(self, game, filepath = None) -> None:
        
        input_shape = game.OBSERVATION_SPACE_VALUES
        action_size = game.ACTION_SPACE_SIZE

        # Neural Net
        # input_layer = Input(shape=input_shape)    # s: batch_size x board_x x board_y

        # hidden_layer = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 2, padding='same')(input_layer)))         # batch_size  x board_x x board_y x num_channels
        # hidden_layer = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 2, padding='same')(hidden_layer)))         # batch_size  x board_x x board_y x num_channels
        # hidden_layer = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 2, padding='same')(hidden_layer)))        # batch_size  x (board_x) x (board_y) x num_channels
        # h_conv4_flat = Flatten()(hidden_layer)       
        # s_fc1 = Dropout(config.DROPOUT_RATE)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        # s_fc2 = Dropout(config.DROPOUT_RATE)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        # policy_head = Dense(action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        # value_head = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        # self.model = Model(inputs=input_layer, outputs=[policy_head, value_head])
        # self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(config.LEARNING_RATE))

        # if filepath is not None:
        #     self.load_model(filepath)
        # Neural Net
        input_layer = Input(shape=input_shape)    # s: batch_size x board_x x board_y

        hidden_layer = Conv2D(512, 2, padding='same')(input_layer)
        hidden_layer = BatchNormalization(axis=3)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        
        hidden_layer = Conv2D(256, 3, padding='same')(input_layer)
        hidden_layer = BatchNormalization(axis=3)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)

        hidden_layer = Flatten()(hidden_layer)       
        
        hidden_layer = Dropout(config.DROPOUT_RATE)(hidden_layer)
        hidden_layer = Dense(1024)(hidden_layer)
        hidden_layer = BatchNormalization(axis=1)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        
        hidden_layer = Dropout(config.DROPOUT_RATE)(hidden_layer)
        hidden_layer = Dense(1024)(hidden_layer)
        hidden_layer = BatchNormalization(axis=1)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)

        hidden_layer = Dropout(config.DROPOUT_RATE)(hidden_layer)
        hidden_layer = Dense(1024)(hidden_layer)
        hidden_layer = BatchNormalization(axis=1)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)

        policy_head = Dense(action_size, activation='softmax', name='pi')(hidden_layer)   # batch_size x self.action_size
        value_head = Dense(1, activation='tanh', name='v')(hidden_layer)                    # batch_size x 1

        self.model = Model(inputs=input_layer, outputs=[policy_head, value_head])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(config.LEARNING_RATE))




    def load_model(self, filepath):
        if os.path.isfile(filepath):
            self.model = load_model(filepath)
    
    def predict(self, game):
        board = game.state
        valid_moves = game.get_valid_actions()
        # print("nn valid moves ", valid_moves, game.board.pieces)
        policy, value = self.model(board)
        policy, value = policy[0], value[0].numpy()[0]

        pi = np.zeros(len(valid_moves))
        for i, move  in enumerate(valid_moves):
            pi[i] = policy[move]
        # print(pi, valid_moves, policy)
        pi = pi/pi.sum()    
        if pi.sum()==0:

            print(pi, valid_moves, policy, board, game.game_finised())
            raise Exception("oldas")
        policy = dict(zip(valid_moves, pi))

        return policy, value
        
    def save_model(self, filepath):
        self.model.save(filepath)

    def train(self, X, y_policy, y_value):
        self.model.fit(
            X,
            [y_policy, y_value],
            batch_size=config.MINIBATCH_SIZE,
            epochs=config.EPOCHS,
        )


        