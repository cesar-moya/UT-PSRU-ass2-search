import numpy as np
import queue
import pytest
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem

class TestSearch:

    def test_game_state_goal_state(self):
        b1 = BoardState()
        gsp = GameStateProblem(b1, b1, 0)

        sln = gsp.search_alg_fnc()
        ref = [(tuple((tuple(b1.state), 0)), None)]

        assert sln == ref

    ## NOTE: If you'd like to test multiple variants of your algorithms, enter their keys below
    ## in the parametrize function. Your set_search_alg should then set the correct method to
    ## use.
    @pytest.mark.parametrize("alg", [""])
    def test_game_state_problem(self, alg):
        """
        Tests search based planning
        """
        b1 = BoardState()
        b2 = BoardState()
        b2.update(0, 14)

        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Single Step
        ref = [(tuple((tuple(b1.state), 0)), (0, 14)), (tuple((tuple(b2.state), 1)), None)]
        assert sln == ref

        b2 = BoardState()
        b2.update(0, 23)
        
        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        ## Two Step:
        ## (0, 14) or (0, 10) -> (any) -> (0, 23) -> (undo any) -> (None, goal state)

        #print(gsp.goal_state_set)
        #print(sln)
        assert len(sln) == 5 ## Player 1 needs to move once, then move the piece back
        assert sln[0] == (tuple((tuple(b1.state), 0)), (0, 14)) or sln[0] == (tuple((tuple(b1.state), 0)), (0, 10))
        assert sln[1][0][1] == 1
        assert sln[2][1] == (0, 23)
        assert sln[4] == (tuple((tuple(b2.state), 0)), None)
    
    # Test Case copied from autograder
    @pytest.mark.parametrize("goal_transform,opt_len", [
        # (((0,14),), 2),
        # (((0,23),), 5), # 1.32s
        # (((0,24),), 6), # 16.53s? (now 40 s)
        # (((1, 7),), 2), # <1s
        # (((1,11),), 2), # <1s
        # (((1,16),), 5), # 1.96s
        (((1,22),(5,22)), 6), # 45s
        # (((1,22),(5,22),(11,50)), 6),
        # (((1,22),(5,22),(6,37),(11,37)), 6), # 47
        # (((4,20),(5,20),(6,22),(11,22)), 7),
    ])
    @pytest.mark.timeout(120)
    def test_search_cases(self, goal_transform, opt_len):
        test_case = (goal_transform, opt_len)
        print(f"\nRunning test case {test_case}")
    
        b1 = BoardState()
        b2 = BoardState()
        # print(f"b1 | {b1.state}")
        # print(f"b2 | {b2.state}")

        for idx, pos in goal_transform:
            b2.update(idx, pos)
        gsp = GameStateProblem(b1, b2, 0)


        # print(f"gsp.b1: {gsp.initial_state}")
        # print(f"gsp.b2: {gsp.goal_state_set}")

        gsp.set_search_alg("")
        sln = gsp.search_alg_fnc()

        assert len(sln) <= opt_len
        
        if test_case == (((0, 14),), 2):
            expected = [
                (tuple((tuple(b1.state), 0)), (0, 14)), 
                (tuple((tuple(b2.state), 1)), None)
            ]
            # assert sln == expected
        elif test_case == (((0,23),), 5):
            expected = [
                # (tuple((tuple(b1.state), 0)), (0, 14)), 
                # (tuple((tuple(b2.state), 1)), None)
            ]
            # assert sln == expected
        else:
            assert True == True
            # print(f"    NO ASSERTION DEFINED!!")
        
        
    

    def test_initial_state(self):
        """
        Confirms the initial state of the game board
        """
        board = BoardState()
        assert board.decode_state == board.make_state()

        ref_state = [(1,0),(2,0),(3,0),(4,0),(5,0),(3,0),(1,7),(2,7),(3,7),(4,7),(5,7),(3,7)]

        assert board.decode_state == ref_state

    def test_generate_actions(self):
        sim = GameSimulator(None)
        generated_actions = sim.generate_valid_actions(0)
        assert len(generated_actions) > 0 # I added this
        assert (0,6) not in generated_actions
        assert (4,0) not in generated_actions

    ## NOTE: You are highly encouraged to add failing test cases here
    ## in order to test your validate_action implementation. To add an
    ## invalid action, fill in the action tuple, the player_idx, the
    ## validity boolean (would be False for invalid actions), and a
    ## unique portion of the descriptive error message that your raised
    ## ValueError should return. For example, if you raised:
    ## ValueError("Cannot divide by zero"), then you would pass some substring
    ## of that description for val_msg.
    @pytest.mark.parametrize("action,player,is_valid,val_msg", [
        ((0,14), 0, True, ""),
        ((0,16), 0, True, ""),
        ((0,10), 0, True, ""),
        ((5,1), 0, True, ""),
        ((5,2), 0, True, ""),
        ((5,4), 0, True, ""),
        ((5,5), 0, True, ""),
        # my test cases
        ((3,9), 0, True, ""),
        ((3,17), 0, True, ""),
        ((3,19), 0, True, ""),
        ((3,13), 0, True, ""),
        ((3,40), 0, False, "invalid"),
        ((2,3), 0, False, "invalid"),
        #black
        ((3,38), 1, True, ""),
        ((3,40), 1, True, ""),
        ((3,44), 1, True, ""),
        ((3,48), 1, True, ""),
        ((3,39), 1, False, "invalid"),
        ((3,17), 1, False, "invalid"),
        ((5,50), 1, True, ""),
        ((5,53), 1, True, ""),
        ((5,52), 1, False, "invalid"),
    ])
    def test_validate_action(self, action, player, is_valid, val_msg):
        sim = GameSimulator(None)
        if is_valid:
            assert sim.validate_action(action, player) == is_valid
        else:
            with pytest.raises(ValueError) as exinfo:
                result = sim.validate_action(action, player)
            assert val_msg in str(exinfo.value)
        

    @pytest.mark.parametrize("state,is_term", [
        ([1,2,3,4,5,3,50,51,52,53,54,52], False), ## Initial State
        ([1,2,3,4,5,55,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,54], True), ## Player 1 wins
        ([1,2,3,4,5,5,50,51,52,53,6,6], True), ## Player 2 wins
        ([1,2,3,4,5,5,50,4,52,53,6,6], False), ## Invalid State
    ])
    def test_termination_state(self, state, is_term):
        board = BoardState()
        board.state = np.array(state)
        board.decode_state = board.make_state()

        assert board.is_termination_state() == is_term

    def test_encoded_decode(self):
        board = BoardState()
        assert board.decode_state  == [board.decode_single_pos(x) for x in board.state]

        enc = np.array([board.encode_single_pos(x) for x in board.decode_state])
        assert np.all(enc == board.state)

    def test_is_valid(self):
        board = BoardState()
        assert board.is_valid()

        ## Out of bounds test
        board.update(0,-1)
        assert not board.is_valid()
        
        board.update(0,0)
        assert board.is_valid()
        
        ## Out of bounds test
        board.update(0,-1)
        board.update(6,56)
        assert not board.is_valid()
        
        ## Overlap test
        board.update(0,0)
        board.update(6,0)
        assert not board.is_valid()

        ## Ball is on index 0
        board.update(5,1)
        board.update(0,1)
        board.update(6,50)
        assert board.is_valid()

        ## Player is not holding the ball
        board.update(5,0)
        assert not board.is_valid()
        
        board.update(5,10)
        assert not board.is_valid()

    @pytest.mark.parametrize("state,reachable,player", [
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(0,1),(2,1),(1,2),(1,0)]), # 1, 7, 15, 9
            0
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(2,2)]), # 16
            1
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(0,0)
            ],
            set(),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,3)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(1,2)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
    ]) 
    def test_ball_reachability(self, state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        # predicted is mine
        # print(f"\nboard.state: {board.state}")
        # print(f"predicted: {predicted_reachable_encoded}")
        assert predicted_reachable_encoded == encoded_reachable

    @pytest.mark.parametrize("state,piece_idx,expected_valid_moves", [
        ([1,2,3,4,5,3,50,51,52,53,54,52], 0, [14, 10, 16]), # piece_idx is array index HERE <----
        ([1,2,3,4,5,3,50,51,52,53,54,52], 4, [18, 10, 20]), # white piece move
        ([1,2,3,4,5,3,50,51,52,53,54,52], 10, [45, 39, 41]), # black piece move
        ([1,2,3,4,5,3,50,51,52,53,54,52], 2, []), # try to move the ball
        ([1,2,3,4,5,3,15,7,52,17,11,52], 1, []), # overlap test, "2" can't move to black occupied spaces
        # ([1,2,3,4,5,3,50,51,52,53,54,52], 10, []), # black piece move
    ])
    def test_single_piece_actions(self, state, piece_idx, expected_valid_moves):
        board = BoardState()
        board.state = state
        board.decode_state = board.make_state()
        actions = Rules.single_piece_actions(board, piece_idx)
        assert sorted(actions) == sorted(expected_valid_moves)
    
    # @pytest.mark.parametrize("state,player_idx,expected_valid_moves", [
    #     ([1,2,3,4,5,3,50,51,52,53,54,52], 0, [1,2,4,5]), # white 
    # ])
    # def test_single_ball_actions(self, state, player_idx, expected_valid_moves):
    #     board = BoardState()
    #     board.state = state
    #     board.decode_state = board.make_state()
    #     actions = Rules.single_ball_actions(board, player_idx)
    #     assert sorted(actions) == sorted(expected_valid_moves)
        