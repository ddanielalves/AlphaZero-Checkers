import re
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin

from Checkers import Checkers as Game
from MCTS import MCTS
from PV_NN import Policy_Value_NN
from Agents import OriginalMCTSAgent as Agent
import config
import dill

app = Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route("/")
def home():
    resp = Response("Foo bar baz")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp




@app.route('/move', methods=['POST'])
@cross_origin()
def move():
    data = request.get_json()

    game = Game(pieces=np.array(data["board"]))
    
    # net = Policy_Value_NN(game, config.BEST_NN_PATH)
    agent = Agent()

    action, prior = agent.predict(game)
    with open("m.pik", "wb") as p:
        dill.dump(agent.MCTS, p)
    return {"action": int(action), "prior": prior}

app.run()

def format_probabilities(prob):
    pass