LIGHT_PIECE_COLOR = "#DDBB7A"
DARK_PIECE_COLOR = "#A52A2A" 

SCREEN_SIZE = (640, 640)
SQUARE_SIZE = 80

C_PUCT =1.4
TAU = 1

REWARD_EATING = 0.3
REWARD_WINNING = 20

NUM_SIMULATIONS = 2000
NUM_RAND_SIMLUATIONS = 0

EPISODES = 200
MAX_QUEUE = 100_000
TRAIN_EVERY = 10
COMPARING_GAMES = 10

BATCH_SIZE = 2056*2
MINIBATCH_SIZE = 128

EPOCHS = 200

# Policy Value Network
BEST_NN_PATH = "./trained_models/best.h5"
LEARNING_RATE = 0.1
DROPOUT_RATE = 0.9
