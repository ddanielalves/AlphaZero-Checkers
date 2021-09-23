"""
Module used to manage the api that exposes the AlphaZero Agents.

Notes:
- In the imports and in the definition of the move api select the Agent to use to answer the queries.

Includes:
- move endpoint: POST method that takes the board as a parameter in the body
- health endpoint: GET method to check if the api can be reached
"""


import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin

from Checkers import Checkers as Game
from PV_NN import Policy_Value_NN

# Select the Agent here and in move api
from Agents import DetermenisticAgent as Agent
# from Agents import OriginalMCTSAgent as Agent


app = Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# @app.route("/")
# def home():
#     resp = Response("Foo bar baz")
#     resp.headers['Access-Control-Allow-Origin'] = '*'
#     return resp

@app.route('/move', methods=['POST'])
@cross_origin()
def move():
    # Get the json object from the body of the request
    data = request.get_json()

    # Create a game object with the current state of game being queried.
    game = Game(nrows= len(data["board"]), pieces=np.array(data["board"]))

    # Load the the best Value_Policy network to be used by the agent.
    # Comment these two lines to select an agent that does not require a nn. 
    net = Policy_Value_NN(game, True)
    agent = Agent(net)
    
    # Uncoment this, together with the OriginalMCTSAgent to use the original MCTS instead of AlphaZero 
    # agent = Agent()

    # predict the current game state
    action, prior = agent.predict(game)

    # Return the action to take and the prior probabilites that the Agent used to select the action. 
    return {"action": int(action), "prior": prior}


@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return "OK", 200

    
app.run()

def format_probabilities(prob):
    pass