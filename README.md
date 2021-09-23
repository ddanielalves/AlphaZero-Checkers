# AlphaZero-Checkers

The AlphaZero algorithm is Deep Reinforcement learning algorithm that combines the Monte Carlo Tree Search with Neural Networks. Before selecting a move the algorithm simulates moves until a never seen or a finished game state is found. Actions are selected based on a policy that balences exploration of low visited states and high value states. The value of a state is queried from the neural network the first time it's reached. These simulations build out the Monte Carlo Tree and focus their exploration in the areas where higher values are expected. For complex game the tree is never finished, therefore the algorithm has predefined number of simulations before it selects a action to take.

This project implements the AlphaZero algorithm in the game of Checkers. Even though checkers is not as complex as chess or go, it is still not a trivial game to solve. To make it easier to validate the algorithm the game was first played on 4x4 board, then on a 6x6 board and finally on a 8x8 board. All these board sizes are valid to play against.

This repository includes:

-   Checkers.py - Module where the game logic is defined.
-   MCTS.py - Module with the logic for the Monte Carlo Tree Search used by the AlphaZero algorithm.
-   MCTS_Original.py - Module with the original MCTS algorithm, in this version after a leaf node is found, instead of querying the NN, a game with random moves is played until the end.
-   Agents.py - Module with the agents required that interact with the end user.
-   PV_NN.py - Module with the definition of the Policy Value Network and the methods required to train it.
-   Trainer - Module responsible for gathering training data and training the model.
-   main_train.py - Script to run the train.
-   main_play.py - Script to play the game using the matplotlib board. It is recommended to use the web interface found in?
-   api.py - API to be used to query the Agent selected via http endpoint.

## Checkers web

This project works best together with the Checkers Web application. The Checkers web application is a web interface to interact with the Agents.
To use the web interface open it on the brower while running this project's api.

In the api.py module can be defined the Agent to answer the http requests.

```
# Select the Agent here and in move api
from Agents import DetermenisticAgent as Agent
# from Agents import OriginalMCTSAgent as Agent
...
def move():
    ...
    # Load the the best Value_Policy network to be used by the agent.
    # Comment these two lines to select an agent that does not require a nn.
    net = Policy_Value_NN(game, True)
    agent = Agent(net)

    # Uncoment this, together with the OriginalMCTSAgent to use the original MCTS instead of AlphaZero
    # agent = Agent()
```

## Setup:

-   git clone https://github.com/ddanielalves/AlphaZero-Checkers.git
-   pip install -r requirements.txt

### To train the model

-   python main_train.py

### To run the api

-   python api.py

### To test the Agents

-   python main_play.py

## References:

https://github.com/junxiaosong/AlphaZero_Gomoku

https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

D. Silver, et al., “Mastering chess and shogi by self-play with a general reinforcement learning algorithm”
