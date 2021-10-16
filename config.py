# Matpltlib configurations
LIGHT_PIECE_COLOR = "#DDBB7A"
DARK_PIECE_COLOR = "#A52A2A" 

SCREEN_SIZE = (640, 640)
SQUARE_SIZE = 80


C_PUCT = 1
TAU = 1

REWARD_WINNING = 1
REWARD_EATING = 0.3

# Number of moves to play between each prediction. Using the NN
NUM_SIMULATIONS = 500
# Number of moves to play between each prediction. Using random moves
NUM_RAND_SIMLUATIONS = 0

# Number of games to train with
EPISODES = 200
# Max size of the training data object
MAX_QUEUE = 100_000
# Number of games to train in between
TRAIN_EVERY = 20
# Number of games to play to test the new best model
COMPARING_GAMES = 10

# Batch size to use to train the NN
BATCH_SIZE = 2056
# Batch size to use in each training epoch of the NN
MINIBATCH_SIZE = 256
# Number of epochs to train the NN
EPOCHS = 200
# Learning rate of the nn train
LEARNING_RATE = 0.1
# Dropout rate of the dropout layers
DROPOUT_RATE = 0.9
