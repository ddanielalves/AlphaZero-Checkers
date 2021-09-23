"""Checkers game logic 
This module represents the board as numpy arrays, this representation makes it easier to access and update the board
however, it is a little slower per iteration.

Includes:
- Board: Object used to represent the checkers board
- Checkers: Object used to represent the Game logic 
"""


# import logging
from threading import active_count
import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from functools import lru_cache
from MCTS import MCTS
import config


DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROWS_TO_PIECES = {8: 12, 6:6, 4:2}


class Board:
    def __init__(self, nrows=8, pieces = None):
        self.nrows = nrows
        
        self.total_pieces = ROWS_TO_PIECES[nrows]
        self.max_piece_nr = int(nrows*nrows/2)
        
        # _board is a matrix with 1 in the valid positions and 0 in the others.
        self._board = np.indices((self.nrows, self.nrows)).sum(axis=0) % 2

        # Flag used to initialize the render figure
        self.ready_to_render = False

        # Array of rows and columns used to convert index to coordinate
        self.board_coords = np.where(self._board == 1)
        self.reset(pieces)


    def draw_piece_p1(self, position):
        """draw player 1 piece on the matplotlib board

        Args:
            position (tuple): coordinates of the piece
        """

        circle = plt.Circle(position, 0.4, color=config.LIGHT_PIECE_COLOR, linewidth=1, ec="white")
        self.board_ax.add_artist(circle)


    def draw_piece_p2(self, position):
        """draw player 2 piece on the matplotlib board

        Args:
            position (tuple): coordinates of the piece
        """

        circle = plt.Circle(position, 0.4, color=config.DARK_PIECE_COLOR, linewidth=1, ec="white")
        self.board_ax.add_artist(circle)
    

    def draw_pieces(self):
        """Draw pieces on the matplotlib board
        """

        Ys, Xs = np.where(self.pieces > 0)
        for i in range(len(Xs)):
            self.draw_piece_p1((Xs[i], Ys[i]))

        Ys, Xs = np.where(self.pieces < 0)
        for i in range(len(Xs)):
            self.draw_piece_p2((Xs[i], Ys[i]))


    def render(self):
        """Render the matplotlib board
        """

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
        """Clear the matplotlib board
        """
        
        self.board_ax.clear()


    def reset(self, pieces=None):
        """Reset the board

        Args:
            pieces (np.array, optional): Array that can be used to set the pieces in the board. 
                                        If None it initializes the board Defaults to None.
        """ 

        if pieces is None:
            self.pieces = np.zeros((self.nrows, self.nrows))

            Xs, Ys = self.board_coords
            for i in range(self.total_pieces):
                self.pieces[Xs[i], Ys[i]] = -1
                self.pieces[Xs[-1 * i - 1], Ys[-1 * i - 1]] = 1
        else:
            self.pieces = pieces


    @lru_cache(maxsize=1000)
    def nr_to_coords(self, nr):
        """Convert the position number to the board coordinate.
        This method is called many times, therefore an lru_cache it is included for performance reasons 

        Args:
            nr (int): Position Number

        Returns:
            tuple: Board coordinates
        """
        return self.board_coords[0][nr], self.board_coords[1][nr]


    def coordinates_to_nr(self, l, c):
        """Convert the board coordinates to a position number

        Args:
            l (int): line/row in the board
            c (int): column in the board

        Returns:
            int: Position number
        """

        for i in range(len(self.board_coords[0])):
            l_, c_ = self.board_coords[0][i], self.board_coords[1][i]
            if l_== l and c_ == c:
                return i
        return -1
    

    def piece_value(self, piece_nr=None, l=None, c=None):
        """Returns the value of the board piece
        Provide piece_nr or l and c

        Args:
            piece_nr (int, optional): Piece Number. Defaults to None.
            l (int, optional): Line/row on the board. Defaults to None.
            c (int, optional): Column on the board. Defaults to None.

        Returns:
            int: Piece value
        """
        if piece_nr is not None:
            l, c = self.nr_to_coords(piece_nr)
        return self.pieces[l, c]


    def set_piece_value(self, value, piece_nr=None, l=None, c=None):
        """Set a value for a piece on the board
        Provide piece_nr or l and c

        Args:
            value (int): Value to set on the board
            piece_nr (int, optional): Piece Number. Defaults to None.
            l (int, optional): Line/row on the board. Defaults to None.
            c (int, optional): Column on the board. Defaults to None.
        """

        if piece_nr is not None:
            l, c = self.nr_to_coords(piece_nr)
        self.pieces[l, c] = value

 
    def reverse(self):
        """Reverse the board so the agents always see it in their perspective
        """
        self.pieces = np.rot90(np.rot90(self.pieces)) * -1


    def move_to_action(self, piece_nr, direction_idx):
        """Convert a move that the game understants into a action that the agent understants

        Args:
            piece_nr (int): Piece Number
            direction_idx (int): index of the DIRECTIONS array defined as a global variable that represents the direction vector.

        Returns:
            int: Action converted
        """

        return piece_nr + direction_idx * self.max_piece_nr


    def action_to_move(self, action):
        """Convert an action to a move

        Args:
            action (int): number from 0 to ACTION_SPACE_SIZE

        Returns:
            piece: int with the piece number
            direction: int with the direction index
        """

        piece = action % self.max_piece_nr
        direction = (action-piece) // self.max_piece_nr
        return piece, direction        


    def check_double_jump(self, piece):
        """Check if the piece can jump above another opponent piece.
        Method invoked afted one jump is made

        Args:
            piece (int): Piece Number

        Returns:
            bool: Flag that represents if another jump is available for the piece
        """

        for direction in range(4):
            action = (piece, direction)
            if self.is_valid_move(action)[1]:
                return True
        return False

    def is_valid_move(self, action):
        """Check if the pair piece-direction is a valid move

        Args:
            action (tuple): Pair piece number-direction

        Returns:
            valid_move: Flag that says if the current move is valid
            jump_available: Flag that says if the current move represents a jump
        """

        if not isinstance(action, tuple):
            action = self.action_to_move(action)

        piece, direction = action
        dir_vector = DIRECTIONS[direction]

        if piece < 0 or piece > self.max_piece_nr:
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
        """Check if the coordinates are inside the board dimensions

        Args:
            l (int): Line/row
            c (int): Column

        Returns:
            bool: Flag to check if the coordinates are inside the board
        """

        if l < 0 or l >= self.nrows:
            return False

        if c < 0 or c >= self.nrows:
            return False

        return True


class Checkers:
    def __init__(self, max_moves=200, pieces = None, nrows=8):
        self.nrows = nrows
        self.board = Board(nrows, pieces=pieces)
        self.max_moves = max_moves
        
        self.first_render = True

        self.ACTION_SPACE_SIZE = nrows * nrows * 2
        self.OBSERVATION_SPACE_VALUES = (1, nrows, nrows)
        self.reset(pieces)

    @property
    def state(self):
        return self.board.pieces.reshape((1, 1, self.nrows, self.nrows))

    def copy(self):
        """Copy the game object deeply, making sure there are no objects copied by reference

        Returns:
            Checkers: Game object copied 
        """

        c = Checkers(self.max_moves, self.board.pieces, nrows=self.board.nrows)
        c.current_move = self.current_move
        c.players_turn = self.players_turn
        c.players_pieces = self.players_pieces.copy()
        c.board.pieces = self.board.pieces.copy()
        return c

    def reset(self, pieces=None):
        """Reset the Checkers game. If pieces are provided use those to initialyze the board

        Args:
            pieces (np.array, optional): Pieces used to initialyze the board. Defaults to None.
        """

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
            self.players_pieces = {1: ROWS_TO_PIECES[self.nrows], -1: ROWS_TO_PIECES[self.nrows]}

        self.board.reset(pieces)


    def render(self):
        """Render the matplotlib board
        """

        self.board.render()

    def get_player_turn(self):
        """Get the player's turn 
        """

        return self.players_turn


    def get_valid_actions(self):
        """Get the valid actions for the current game state

        Returns:
            list: list of valid actions 
        """

        valid_moves = self.get_valid_moves()
        actions = []

        for move in valid_moves:
            action = self.board.move_to_action(*move)
            actions.append(action)
            
        return actions


    def get_valid_moves(self):
        """Compute the list of valid moves. 
        Similar to the list of valid actions but here the output is a pair of piece number - direction.

        Returns:
            list: list of valid moves
        """

        if self.valid_moves_updated:
            return self.valid_moves

        valid_moves = []
        jump_moves = []
        directions = [0, 1, 2, 3]

        for piece in range(self.board.max_piece_nr):
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
        """Update the board with the action provided 

        Args:
            action (int): 

        Raises:
            Exception: If action is not valid raise an exception for debugging purposes
        """

        if not isinstance(action, tuple):
            old_a = action
            action = self.board.action_to_move(action)

        valid, jump = self.board.is_valid_move(action)
        if not valid:
            raise Exception("not valid", self.board.pieces, old_a, action, self.get_valid_moves(), self.players_turn )

        piece, dir_vector = action
        if not isinstance(dir_vector, tuple):
            dir_vector = DIRECTIONS[dir_vector]
        piece_val = self.board.piece_value(piece)
                
        self.board.set_piece_value(0, piece)

        p_l, p_c = self.board.nr_to_coords(piece)

        m_l, m_c = p_l + dir_vector[0], p_c + dir_vector[1]

        if jump:
            # Set the piece value to 0 on the board
            self.board.set_piece_value(0, c=m_c, l=m_l)
            self.players_pieces[-self.players_turn] -= 1
            # Adding one direction vector to the position where the piece will land
            m_l, m_c = m_l + dir_vector[0], m_c + dir_vector[1]

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


    def step(self, action):
        """Method used to interact with the agents and the RL models

        Args:
            action (int): Action to take

        Returns:
            done: If the game has ended
            winner: The winner if there is one
            players_turn: the player to move now
        """

        self.current_move += 1
        self.play(action)
        done, winner = self.game_finished()
        return done, winner, self.players_turn
            

    def game_finished(self):
        """Check if the game has finished

        Returns:
            done: Flag if there is the game has finished
            winner: 1 or -1 if player 1 or -1 has won, 0 if there is a draw, else None
        """
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
