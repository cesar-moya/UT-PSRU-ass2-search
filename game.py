import numpy as np
import queue

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        c, r = cr
        res = (6 * r) + r + c
        # print(f"encode: ({c},{r}) = {res}")
        return int(res)

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        c = int(n % 7) # the remainder is the column
        r = int(n // 7) # the quotient is the row
        # print(f"decode: {n} = ({c}, {r})")
        return (c, r)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.
        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        """
        if not self.is_valid():
            return False

        # TODO: can we assume that white always start in the 0 row?
        w_blocks, w_ball = self.get_white()
        b_blocks, b_ball = self.get_black()
        
        # white wins?
        c, r = self.decode_single_pos(w_ball)
        if r == self.N_ROWS-1:
            return True

        # black wins?
        c, r = self.decode_single_pos(b_ball)
        if r == 0:
            return True

        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.
        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.
        Output: return True (if valid) or False (if not valid)
        """
        # print("State: ", self.state)
        # length and basic checks for valid numbers
        if len(self.state) != 12:
            return False
        for s in self.state:
            if s < 0 or s > 55:
                return False
        # check ball is on a correct piece
        w_blocks, w_ball = self.get_white()
        b_blocks, b_ball = self.get_black()
        # print(f"white: {w_blocks}, w_ball: {w_ball} | black: {b_blocks}, b_ball: {b_ball}")
        if w_ball not in w_blocks or b_ball not in b_blocks:
            # print("some ball is floating in space")
            return False

        # check pieces are not overlapping
        # all_blocks = w_blocks + b_blocks
        all_blocks = [w for w in w_blocks] + [b for b in b_blocks]
        #print(f"all_blocks: {all_blocks}")
        if len(all_blocks) != len(set(all_blocks)):
            # print("overlapping pieces found")
            return False
        return True

    def get_white(self):
        w_blocks = self.state[:5]
        w_ball = self.state[5]
        return (w_blocks, w_ball)
        
    def get_black(self):
        b_blocks = self.state[6:11]
        b_ball = self.state[11]
        return (b_blocks, b_ball)
    
    def same_diagonal(self, x1, y1, x2, y2):
        return abs(x2 - x1) == abs(y2 - y1)
            
    # sub-function that finds all friendly blocks that are reachable (direct line of sight)
    def get_reachable_neighbors(self, current_block, friendly_blocks, opposite_blocks):
        # loop through friendly_blocks, check if it's in view and without any opposite in the way
        reachable = []
        for f in friendly_blocks:
            if f != current_block and self.is_reachable(current_block, f, opposite_blocks):
                # print(f"** FROM {current_block} TARGET {f} is REACHABLE **")
                reachable.append(f)
        return reachable

    # Checks whether the target block is reachable (in clear sight) from current (horizontal, diagonal, vertical)
    def is_reachable(self, current_block, target_block, opposite_blocks):
        if current_block == target_block:
            return False
        
        curr_c, curr_r = self.decode_single_pos(current_block)
        targ_c, targ_r = self.decode_single_pos(target_block)
        # a friendly block is a candidate for taking the ball if it's on either the same col, same row, or same diagonal
        is_candidate = curr_c == targ_c or curr_r == targ_r or self.same_diagonal(curr_c, curr_r, targ_c, targ_r)
        # print(f"\nChecking... | curr:{current_block} | targ: {target_block} | is_candidate: {is_candidate}")
        if not is_candidate:
            return False

        opposite_blocks_dec = [self.decode_single_pos(s) for s in opposite_blocks]
        # print(f"Check Obstructions | opposite_blocks: {opposite_blocks}")
        # if target piece is on same col, check if there are blocking blocks
        if curr_c == targ_c:
            for c, r in opposite_blocks_dec:
                if curr_c == c and min(curr_r, targ_r) <= r <= max(curr_r, targ_r):
                    # print(f"        COL Collision")
                    return False
        
        # if target piece is on same row, check if there are blocking blocks
        if curr_r == targ_r:
            for c, r in opposite_blocks_dec:
                if curr_r == r and min(curr_c, targ_c) <= c <= max(curr_c, targ_c):
                    # print(f"        ROW Collision")
                    return False
        
        # if target piece is on same diagonal, check if there are any blocking blocks
        if self.same_diagonal(curr_c, curr_r, targ_c, targ_r):
            for opp_c, opp_r in opposite_blocks_dec:
                is_blocking = ( min(curr_c, targ_c) < opp_c < max(curr_c, targ_c) and 
                                min(curr_r, targ_r) < opp_r < max(curr_r, targ_r) )
                if is_blocking:
                    # print(f"        DIAG Collision")
                    return False
        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        """
        piece_enc = board_state.state[piece_idx] # the encoded location of the piece, e.g. "53"
        (c, r) = board_state.decode_single_pos(piece_enc)
        # print(f"\nstate: {board_state.state} | piece_idx: {piece_idx} | piece_enc: {piece_enc} | c, r: {(c, r)}")
        # these are encoded
        w_blocks, w_ball = board_state.get_white()
        b_blocks, b_ball = board_state.get_black()
        if piece_enc == w_ball or piece_enc == b_ball:
            return []
        # jump top-left
        top_left_far =  (c - 1, r + 2)
        top_left_near = (c - 2, r + 1)
        # jump top-right
        top_right_far =  (c + 1, r + 2)
        top_right_near = (c + 2, r + 1)
        # jump botton-left
        bottom_left_far =  (c - 1, r - 2)
        bottom_left_near = (c - 2, r - 1)
        # jump bottom-right
        bottom_right_far =  (c + 1, r - 2)
        bottom_right_near = (c + 2, r - 1)
        options = [top_left_far, top_left_near, top_right_far, top_right_near, 
                   bottom_left_far, bottom_left_near, bottom_right_far, bottom_right_near]
        valid_moves = []
        for o in options:
            col, row = o
            # check out of bounds
            if col < 0 or col > board_state.N_COLS-1 or row < 0 or row > board_state.N_ROWS-1:
                continue 
            # check if target cell is occupied
            o_enc = board_state.encode_single_pos(o)
            if o_enc in w_blocks or o_enc in b_blocks:
                continue # cell occupied already
            valid_moves.append(o_enc)
        # print(f"valid_moves: {valid_moves}")
        return valid_moves

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx in the board_state
        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        """
        w_blocks, w_ball = board_state.get_white()
        b_blocks, b_ball = board_state.get_black()
        # print(f"\nstate: {board_state.state} | player: {player_idx} | w_blocks: {w_blocks} | b_blocks: {b_blocks}")
        # the ball can only move to one of its blocks in clear sight at any distance like a queen
        if player_idx == 0: # white
            ball_valid_actions = Rules.get_ball_actions_BFS(board_state, w_ball, w_blocks, b_blocks)    
        elif player_idx == 1: # black
            ball_valid_actions = Rules.get_ball_actions_BFS(board_state, b_ball, b_blocks, w_blocks)
        else:
            raise Exception("Invalid Player")
        return ball_valid_actions
    
    # blocks and ball are encoded (e.g. 53, 2, 27)
    @staticmethod
    def get_ball_actions_BFS(board_state: BoardState, ball, friendly_blocks, opposite_blocks) -> set:
        visited = set()
        q = queue.SimpleQueue()
        q.put(ball)
        while not q.empty():
            block = q.get()
            # don't return the ball's block as an option
            if block != ball:
                visited.add(block)
            for n in board_state.get_reachable_neighbors(block, friendly_blocks, opposite_blocks):
                if n not in visited:
                    q.put(n)
        return {int(x) for x in visited} # for clarity

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
        """
        # Rules.single_piece_actions(board_state, piece_idx)
        # Rules.single_ball_actions (board_state, player_idx)
        if player_idx == 0:
            blocks, ball = self.game_state.get_white()
        else:
            blocks, ball = self.game_state.get_black()

        actions = set()
        for piece_idx, pos_enc in enumerate(blocks):
            piece_actions = Rules.single_piece_actions(self.game_state, piece_idx)
            actions.add( (piece_idx, action) for action in piece_actions )

        ball_actions = Rules.single_ball_actions(self.game_state, player_idx)
        actions.add( (piece_idx, action) for action in ball_actions )
        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """
        if False:
            raise ValueError("For each case that an action is not valid, specify the reason that the action is not valid in this ValueError.")
        if True:
            return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
