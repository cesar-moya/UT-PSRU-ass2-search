import math
import numpy as np
import queue, heapq
from game import BoardState, GameSimulator, Rules
import collections

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([
                        tuple(
                            (tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1)
                        )
                    ])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), 
                         set([
                             tuple((tuple(goal_board_state.state), 0)), 
                             tuple((tuple(goal_board_state.state), 1))
                        ]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()
        self.goal_board = goal_board_state.state

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        if alg == "bfs":
            self.search_alg_fnc = self.moya_search
        elif alg == "bfs_deque":
            self.search_alg_fnc = self.moya_search_deque
        elif alg == "bfs_heapq":
            self.search_alg_fnc = self.moya_search_heapq
        elif alg == "dijkstra_heapq":
            self.search_alg_fnc = self.moya_search_dijkstra_heapq
        elif alg == "dijkstra":
            self.search_alg_fnc = self.moya_search_dijkstra
        else:
            # default
            self.search_alg_fnc = self.moya_search_dijkstra_heapq
            alg = "dijkstra_heapq"

        print(f"Now using search algorighm: {alg}")
        

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action
        The action is given with respect to the current player
        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))
    
    def moya_search(self):
        # state = (encoded_state, player_idx)
        print(f"\ninitial_state: {self.state_str(self.initial_state)}")
        myQueue = queue.SimpleQueue()
        myQueue.put(self.initial_state)
        visited = {}
        visited[self.initial_state] = True
        parent = {}
        actions_dequeued = 0
        actions_enqueued = 0
        actions_potential = 0
        while not myQueue.empty():
            actions_dequeued += 1
            state = myQueue.get()
            if self.is_goal(state):
                break
            new_actions = self.get_actions(state)
            # action = (relative_idx, encoded_position);  relative_idx = {0..5}, enc_pos = {0...58~}
            for a in new_actions:
                actions_potential += 1
                parent_state_action = (state, a)
                new_state = self.execute(state, a)
                if new_state not in visited:
                    actions_enqueued += 1
                    visited[new_state] = True
                    myQueue.put(new_state)
                    parent[new_state] = parent_state_action

        solution = self.extract_solution(parent, state)
        self.print_solution(solution, actions_dequeued, actions_enqueued, actions_potential)
        return solution

    def moya_search_deque(self):
        # state = (encoded_state, player_idx)
        print(f"\ninitial_state: {self.state_str(self.initial_state)}")
        deque = collections.deque()
        deque.append(self.initial_state)
        # myQueue.put(self.initial_state)
        visited = {}
        visited[self.initial_state] = True
        parent = {}
        actions_dequeued = 0
        actions_enqueued = 0
        actions_potential = 0
        while len(deque) > 0:
            actions_dequeued += 1
            state = deque.popleft()
            if self.is_goal(state):
                break
            new_actions = self.get_actions(state)
            # action = (relative_idx, encoded_position);  relative_idx = {0..5}, enc_pos = {0...58~}
            for a in new_actions:
                actions_potential += 1
                parent_state_action = (state, a)
                new_state = self.execute(state, a)
                if new_state not in visited:
                    actions_enqueued += 1
                    visited[new_state] = True
                    deque.append(new_state)
                    parent[new_state] = parent_state_action

        solution = self.extract_solution(parent, state)
        self.print_solution(solution, actions_dequeued, actions_enqueued, actions_potential)
        return solution

    def moya_search_heapq(self):
        # state = (encoded_state, player_idx)
        print(f"\ninitial_state: {self.state_str(self.initial_state)}")
        heap = []
        #myQueue.put(self.initial_state)
        heapq.heappush(heap, (0, self.initial_state))
        visited = {}
        visited[self.initial_state] = True
        parent = {}
        actions_dequeued = 0
        actions_enqueued = 0
        actions_potential = 0
        while len(heap) > 0:
            actions_dequeued += 1
            h, state = heapq.heappop(heap)
            if self.is_goal(state):
                break
            new_actions = self.get_actions(state)
            # action = (relative_idx, encoded_position);  relative_idx = {0..5}, enc_pos = {0...58~}
            for a in new_actions:
                actions_potential += 1
                parent_state_action = (state, a)
                new_state = self.execute(state, a)
                if new_state not in visited:
                    actions_enqueued += 1
                    visited[new_state] = True
                    heapq.heappush(heap, (actions_dequeued, new_state))
                    parent[new_state] = parent_state_action

        solution = self.extract_solution(parent, state)
        self.print_solution(solution, actions_dequeued, actions_enqueued, actions_potential)
        return solution

    def moya_search_dijkstra(self):
        # state = (encoded_state, player_idx)
        print(f"\ninitial_state: {self.state_str(self.initial_state)}")
        myQueue = queue.PriorityQueue()
        myQueue.put((0, self.initial_state))
        cost = {}
        cost[self.initial_state] = 0
        parent = {}
        actions_dequeued = 0
        actions_enqueued = 0
        actions_potential = 0
        while not myQueue.empty():
            actions_dequeued += 1
            h, current = myQueue.get()
            if self.is_goal(current):
                break
            new_actions = self.get_actions(current)
            # action = (relative_idx, encoded_position);  relative_idx = {0..5}, enc_pos = {0...58~}
            for a in new_actions:
                actions_potential += 1
                parent_state_action = (current, a)
                next = self.execute(current, a)
                h = self.heuristic(next)
                new_cost = cost[current] + h
                if next not in cost or new_cost < cost[next]:
                    actions_enqueued += 1
                    cost[next] = new_cost
                    
                    myQueue.put((new_cost, next))  # or should it be just h?
                    parent[next] = parent_state_action

        solution = self.extract_solution(parent, current)
        self.print_solution(solution, actions_dequeued, actions_enqueued, actions_potential)
        return solution

    def moya_search_dijkstra_heapq(self):
        # state = (encoded_state, player_idx)
        print(f"\ninitial_state: {self.state_str(self.initial_state)}")
        heap = []
        heapq.heappush(heap, (0, self.initial_state))
        cost = {}
        cost[self.initial_state] = 0
        parent = {}
        actions_dequeued = 0
        actions_enqueued = 0
        actions_potential = 0
        while len(heap) > 0:
            actions_dequeued += 1
            h, current = heapq.heappop(heap)
            if self.is_goal(current):
                break
            new_actions = self.get_actions(current)
            # action = (relative_idx, encoded_position);  relative_idx = {0..5}, enc_pos = {0...58~}
            for a in new_actions:
                actions_potential += 1
                parent_state_action = (current, a)
                next = self.execute(current, a)
                h = self.heuristic(next)
                new_cost = cost[current] + h
                if next not in cost or new_cost < cost[next]:
                    actions_enqueued += 1
                    cost[next] = new_cost
                    heapq.heappush(heap, (new_cost, next))  # or should it be just h?
                    parent[next] = parent_state_action

        solution = self.extract_solution(parent, current)
        self.print_solution(solution, actions_dequeued, actions_enqueued, actions_potential)
        return solution

    def heuristic(self, state):
        return 1
        # state_helper = BoardState()
        # total_cost = 0
        # board_state, player = state
        # # maybe take a heuristic only for the current player, leave the rest intact?
        # for i, val in enumerate(board_state):
        #     p1 = state_helper.decode_single_pos(val)
        #     p2 = state_helper.decode_single_pos(self.goal_board[i])

        #     if i == 5 or i == 11: #skip the ball for now
        #         # total_cost += self.ball_distance_heuristic(p1, p2)
        #         pass
        #     else:
        #         # total_cost += self.euclidean_distance(p1, p2)
        #         # total_cost += self.manhattan_distance(p1, p2)
        #         total_cost += self.knight_dist_heuristic(p1, p2)
        # return total_cost
    
    def ball_distance_heuristic(self, ball_pos, goal_pos):
        knight_distance = self.knight_dist_heuristic(ball_pos, goal_pos)

        # (Simplified) Adjustment for queen-like movement (replace with more robust logic)
        if abs(ball_pos[0] - goal_pos[0]) == abs(ball_pos[1] - goal_pos[1]):  # Diagonal
            knight_distance = math.ceil(knight_distance / 2)  # Reduce for diagonal

        return knight_distance 

    def knight_dist_heuristic(self, p1, p2):
        x_dist = abs(p1[0] - p2[0])
        y_dist = abs(p1[1] - p2[1])
        return max(math.ceil(x_dist / 3), math.ceil(y_dist / 3))
    
    def manhattan_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)
    
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def extract_solution(self, parent, last_state):
        # print(f"last_state: {self.state_str(last_state)}")
        solution = []
        solution.append((last_state, None))
        while last_state in parent: 
            actionState = parent[last_state]
            if actionState is not None:
                s, a = actionState
                last_state = s
                solution.append((s, a))
            else:
                break

        solution.reverse()
        return solution
    
    def print_solution(self, solution, actions_dequeued, actions_enqueued, actions_potential):
        print(f"Solution: ")
        for i in solution:
            state, action = i
            board_state, player = state
            state_str = ", ".join(str(int(x)) for x in board_state)
            action_str = (int(action[0]), int(action[1])) if action is not None else " None "
            actual_bfs = Rules.total_bfs_searches - Rules.total_bfs_avoided
            print(f"state: {state_str} | player {player} | action: {action_str} | actions_dequeued: {actions_dequeued:,} | "
                  f"actions_enqueued: {actions_enqueued:,} | actions_potential: {actions_potential:,} | "
                  f"ball_BFS_search_attempts: {Rules.total_bfs_searches:,} | actual_ball_BFS_searches: {actual_bfs:,}"
                  , end="\n")

    def state_str(self, state):
        st, pl = state
        return "(" + ", ".join(str(int(s)) for s in st) + f") | player {pl}"


    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    # """ Here is an example:
    # def my_snazzy_search_algorithm(self):
    #     ## Some kind of search algorithm
    #     ## ...
    #     return solution ## Solution is an ordered list of (s,a)
    # """