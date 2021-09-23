import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Activation, Dropout, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np

import config

class Policy_Value_NN:
    def __init__(self, game, load_best = False) -> None:
        """Definition of the layers for the Policy Value NN. 
        The network has one input layer and two output layers, one for the value and other for the policy.

        Args:
            game (Game): game object to obtain the observation and action space sizes.
            load_best (bool, optional): Flag that defined if the best nn should be loaded from memory. Defaults to False.
        """

        input_shape = game.OBSERVATION_SPACE_VALUES
        action_size = game.ACTION_SPACE_SIZE

        # Neural Net
        input_layer = Input(shape=input_shape)

        hidden_layer = Conv2D(512, 2, padding='same')(input_layer)
        hidden_layer = BatchNormalization(axis=3)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        
        hidden_layer = Conv2D(256, 3, padding='same')(input_layer)
        hidden_layer = BatchNormalization(axis=3)(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)

        hidden_layer = Flatten()(hidden_layer)       
        
        hidden_layer = Concatenate()([hidden_layer, Flatten()(input_layer)])

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
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(config.LEARNING_RATE))

        if load_best:
            self.load_model(f"./trained_models/best_{game.nrows}.h5")


    def load_model(self, filepath):
        """Load a model from the disk

        Args:
            filepath (str): filepath to the model to load.
        """

        if os.path.isfile(filepath):
            self.model = load_model(filepath)
    
    def predict(self, game):
        """Predict the policy and value for the current game state

        Args:
            game (Game): Game with the board state to predict

        Raises:
            Exception: Raise an exception if the network outputs a policy that does not match the valid moves for debug

        Returns:
            policy: dict of moves and probabilities
            value: value the nn gives to the state
        """

        board = game.state
        valid_moves = game.get_valid_actions()

        policy, value = self.model(board)
        policy, value = policy[0], value[0].numpy()[0]

        pi = np.zeros(len(valid_moves))
        for i, move  in enumerate(valid_moves):
            pi[i] = policy[move]

        if pi.sum() == 0:
            print(pi, valid_moves, policy, board, game.game_finished())
            raise Exception("policy summing to 0")
            
        pi = pi/pi.sum()    
        policy = dict(zip(valid_moves, pi))

        return policy, value
        
    def save_model(self, filepath):
        """Saving the model to disk

        Args:
            filepath (str): filepath to place the model
        """
        self.model.save(filepath)

    def train(self, X, y_policy, y_value):
        self.model.fit(
            X,
            [y_policy, y_value],
            batch_size=config.MINIBATCH_SIZE,
            epochs=config.EPOCHS,
        )


        