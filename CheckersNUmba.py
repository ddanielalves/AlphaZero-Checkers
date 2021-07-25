# import logging
from threading import active_count
import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from functools import lru_cache
from MCTS import MCTS
import config
from numba import njit


DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

_board = np.indices((8, 8)).sum(axis=0) % 2
board_coords = np.where(_board == 1)

class Board:
    def __init__(self, pieces = None):
        self.nrows = 8
        
        # _board is a matrix with 1 in the valid positions and 0 in the others.
        

        # Flag used to initialize the render figure
        self.ready_to_render = False

        # Array of rows and columns used to convert index to coordinate
        
        self.reset(pieces)

    def reset(self, pieces=None):
        if pieces is None:
            self.pieces = np.zeros((self.nrows, self.nrows))

            Xs, Ys = self.board_coords
            for i in range(12):
                self.pieces[Xs[i], Ys[i]] = -1
                self.pieces[Xs[-1 * i - 1], Ys[-1 * i - 1]] = 1
        else:
            self.pieces = pieces

        # state_ = self.pieces.reshape((1,8,8))

    @lru_cache(maxsize=1000)
    def nr_to_coords(self, nr):
        return self.board_coords[0][nr], self.board_coords[1][nr]

    def coordinates_to_nr(self, l, c):
        for i in range(len(self.board_coords[0])):
            l_, c_ = self.board_coords[0][i], self.board_coords[1][i]
            if l_== l and c_ == c:
                return i
        return -1
    
    def piece_value(self, piece_nr=None, l=None, c=None):
        if piece_nr is not None:
            l, c = self.nr_to_coords(piece_nr)
        return self.pieces[l, c]

    def set_piece_value(self, value, piece_nr=None, l=None, c=None):
        if piece_nr is not None:
            l, c = self.nr_to_coords(piece_nr)
        self.pieces[l, c] = value

    def draw_piece_p1(self, position):
        circle = plt.Circle(position, 0.4, color=config.LIGHT_PIECE_COLOR, linewidth=1, ec="white")
        self.board_ax.add_artist(circle)

    def draw_piece_p2(self, position):
        circle = plt.Circle(position, 0.4, color=config.DARK_PIECE_COLOR, linewidth=1, ec="white")
        self.board_ax.add_artist(circle)
    
    def draw_pieces(self):
        Ys, Xs = np.where(np.array(self.pieces) > 0)
        for i in range(len(Xs)):
            self.draw_piece_p1((Xs[i], Ys[i]))

        Ys, Xs = np.where(np.array(self.pieces) < 0)
        for i in range(len(Xs)):
            self.draw_piece_p2((Xs[i], Ys[i]))


    def reverse(self):
        self.pieces = np.rot90(np.rot90(self.pieces)) * -1

    def render(self):
        if self.ready_to_render == False:
            self.fig, self.board_ax = plt.subplots()
            self.ready_to_render = True

        self.clear_board()
        self.board_ax.matshow(self._board, cmap="Greys")

        self.draw_pieces()

        Ys, Xs = self.board_coords
        for i, _ in enumerate(Xs):
            txt = self.board_ax.text(Xs[i] - 0.15, Ys[i] + 0.1, str(i), color="white", fontsize=10)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block = False)


    def clear_board(self):
        self.board_ax.clear()

    @staticmethod
    def move_to_action(piece_nr, direction):
        return piece_nr + direction * 32

    def action_to_move(self, action):
        piece = action % 32
        direction = (action-piece) // 32
        return piece, direction        


    def check_double_jump(self, piece):
        for direction in range(4):
            action = (piece, direction)
            if self.is_valid_move(action)[1]:
                return True
        return False

    def is_valid_move(self, action):
        if not isinstance(action, tuple):
            action = self.action_to_move(action)

        piece, direction = action
        dir_vector = DIRECTIONS[direction]

        if piece < 0 or piece > 32:
            return False, False

        if direction < 0 or direction > 4:
            return False, False

        piece_value = self.piece_value(piece)

        if piece_value <= 0:
            return False, False

        king = piece_value == 2
        if dir_vector[0] == 1 and not king:
            return False, False

        l, c = self.nr_to_coords(piece)

        l_m, c_m = l + dir_vector[0], c + dir_vector[1]
        if not self.valid_position(l_m, c_m):

            return False, False

        move_value = self.piece_value(l=l_m, c=c_m)
        if move_value == 0:
            return True, False

        if move_value > 0:
            return False, False

        l_m, c_m = l_m + dir_vector[0], c_m + dir_vector[1]
        if not self.valid_position(l_m, c_m):
            return False, False

        move_value = self.piece_value(l=l_m, c=c_m)

        if move_value == 0:
            return True, True

        return False, False

    def valid_position(self, l, c):
        if l < 0 or l >= 8:
            return False

        if c < 0 or c >= 8:
            return False

        return True


class Checkers:
    def __init__(self, max_moves=500, pieces = None):

        self.board = Board(pieces=pieces)
        self.max_moves = max_moves
        
        self.first_render = True

        self.ACTION_SPACE_SIZE = 32 * 4
        self.OBSERVATION_SPACE_VALUES = (1, 8, 8)
        self.reset(pieces)

    @property
    def state(self):
        return self.board.pieces.reshape((1, 1, 8, 8))

    def copy(self):
        c = Checkers(self.max_moves, self.board.pieces)
        c.current_move = self.current_move
        c.players_turn = self.players_turn
        c.players_pieces = self.players_pieces.copy()
        c.board.pieces = self.board.pieces.copy()
        return c

    def reset(self, pieces=None):
        self.done = False
        self.current_move = 0
        self.valid_moves_updated = False

        # 1 if its player one's turn or -1 if it's player two's turn
        # players_turn = 1 - players_turn  
        self.players_turn = 1

        # Nr of pieces of each player
        if pieces is not None:
            p1 = len(np.where(pieces >= 1)[0])
            p2 = len(np.where(pieces <= -1)[0])
            self.players_pieces = {1: p1, -1: p2}

        else:
            self.players_pieces = {1: 12, -1: 12}

        return self.board.reset(pieces)


    def get_valid_actions(self):
        valid_moves = self.get_valid_moves()
        actions = []

        for move in valid_moves:
            action = Board.move_to_action(*move)
            actions.append(action)
            
        return actions


    def render(self):
        self.board.render()

    def get_player_turn(self):
        return self.players_turn

    def get_valid_moves(self):


        if self.valid_moves_updated:
            return self.valid_moves

        valid_moves = []
        jump_moves = []
        directions = [0, 1, 2, 3]

        for piece in range(32):
            for direction in directions:
                action = (piece, direction)
                valid, jump = self.board.is_valid_move(action)

                if valid:
                    if jump:
                        jump_moves.append(action)
                    else:
                        valid_moves.append(action)

        if len(jump_moves) > 0:
            self.valid_moves = jump_moves
        else:     
            self.valid_moves = valid_moves
        
        self.valid_moves_updated = True
        return self.valid_moves

 
    def play(self, action):
        if not isinstance(action, tuple):
            old_a = action
            action = self.board.action_to_move(action)

        valid, jump = self.board.is_valid_move(action)
        if not valid:
            raise Exception("notvalid", self.board.pieces, old_a, action, self.get_valid_moves(), self.players_turn )

        piece, dir_vector = action
        if not isinstance(dir_vector, tuple):
            dir_vector = DIRECTIONS[dir_vector]
        piece_val = self.board.piece_value(piece)
        
        reward = 0        
        
        self.board.set_piece_value(0, piece)

        p_l, p_c = self.board.nr_to_coords(piece)

        m_l, m_c = p_l + dir_vector[0], p_c + dir_vector[1]

        if jump:
            # Set the piece value to 0 on the board
            self.board.set_piece_value(0, c=m_c, l=m_l)
            self.players_pieces[-self.players_turn] -= 1
            # Adding one direction vector to the position where the piece will land
            m_l, m_c = m_l + dir_vector[0], m_c + dir_vector[1]

            reward = config.REWARD_EATING
        # If it's not a jump or if it is a jump without the possibility of a double jump.
        # Change the turn to the other player
        if m_l == 0:
            piece_val = 2
        self.board.set_piece_value(piece_val, c=m_c, l=m_l)

        piece = self.board.coordinates_to_nr(m_l, m_c)
        if not (jump and self.board.check_double_jump(piece)):
            self.players_turn = -self.players_turn
            self.board.reverse()

        self.valid_moves_updated = False

        return reward

    def step(self, action):
        self.current_move += 1
        reward = self.play(action)
        done, winner = self.game_finished()
        return done, winner, self.players_turn, reward
            

    def game_finished(self):
        done = False
        winner = None
        
        # Check if player 2 is out o pieces
        if self.players_pieces[-1] == 0:
            done = True
            winner = 1

        # Check if player 1 is out o pieces
        elif self.players_pieces[1] == 0:
            done = True
            winner = -1

        # Check for draw
        elif self.current_move > self.max_moves:
            done = True
            winner = 0
        
        # Check if the current player is unable to move
        elif len(self.get_valid_moves()) == 0:
            done = True 
            winner = -self.players_turn

        return done, winner
