import re
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin

from Checkers import Checkers as Game
from MCTS import MCTS
from PV_NN import Policy_Value_NN
from Agents import StochasticAgent as Agent
import config

app = Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


@app.route('/move', methods=['POST'])
@cross_origin()
def move():
    data = request.get_json()

    game = Game()
    game.board.pieces = np.array(data["board"])
    net = Policy_Value_NN(game, config.BEST_NN_PATH)
    agent = Agent(net)

    action, prior = agent.predict(game)
    print(prior)

    return {"action": int(action), "prior": prior}

app.run()

def format_probabilities(prob):
    pass