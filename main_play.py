from Agents import HumanAgent
from Agents import StochasticAgent as OpponentAgent
from Checkers import Checkers
from PV_NN import Policy_Value_NN


def move_mcts_roots(p1,p2, action):
    print("moving to action", action)
    if p1.has_MCTS:
        p1.MCTS.move_root(action)
    
    if p2.has_MCTS:
        p2.MCTS.move_root(action)


if __name__ == "__main__":    
    game = Checkers()

    p1 = HumanAgent()

    pv_net = Policy_Value_NN(game)
    p2 = OpponentAgent(pv_net)
    
    players = {1: p1, -1:p2}

    done = False
    state = game.reset()
    players_turn = 1

    while not done:
        if players_turn == 1:
            game.render()
        
        action = players[players_turn].predict(game)[0]
        print(action)
        done, winner, players_turn = game.step(action)
        
        move_mcts_roots(p1,p2, action)
