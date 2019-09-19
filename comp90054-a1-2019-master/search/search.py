# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):

    from util import Stack
    from game import Directions

    # initial open and closed lists
    open = Stack()
    closed = []
    # get start state into the open list
    open.push((problem.getStartState(), [], []))

    while not open.isEmpty():
        for limit in range(100):
            current_node, actions, depth = open.pop()
            '''print("actions:", actions)
            print("current_node:", current_node)
            print("depth", depth)'''
            # if current state is the goal state
            # return list of actions
            if problem.isGoalState(current_node):
                return actions

            if current_node not in closed:
                # expand current node
                # add current node to closed list
                expand = problem.getSuccessors(current_node)
                # print("expand:", expand)
                closed.append(current_node)
                # print("closed:", closed)

                for location, direction, cost in expand:
                    # if the location has not been visited, put into open list
                    if (location not in closed):
                        open.push((location, actions + [direction], limit))

    ''' 
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    import copy
    from util import Stack

    if stack == None:
        stack = util.Stack()
        explored.append(problem.getStartState())
        start_node = [problem.getStartState(), []]
        stack.push(start_node)

    cur_node = stack.pop()
    #print(cur_node)
    #print("explored:", explored)
    next_successor = problem.getSuccessors(cur_node[0])
    #print("next successor:", next_successor)
    #print("next_pos:")

    for successor in next_successor:
        # if successor[0] not in explored and successor[0] not in expand:
        if successor[0] not in explored:
            # print("cur_node:",cur_node)
            #print(successor[0])
            #print(successor[1])
            move = []

            for action in cur_node[1]:
                move.append(action)
            move.append(successor[1])
            #print("move:", move)
            #print("successor:", successor[1])
            if problem.isGoalState(successor[0]):
                return move
            new_node = [successor[0], move]
            stack.push(new_node)
            explored.append(successor[0])
            # print(successor[0])
            if depth+1 < limit:
                depth_limit = depthFirstSearch(problem, explored, stack, depth+1, limit)
                if depth_limit is not None:
                    return depth_limit

'''
def depthFirstSearchLimit(problem, closed = None, stack = None, depth = 1, limit = 100):
    '''print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))'''

    if stack is None:
        stack = util.Stack()
        solution = []
        start_node = [problem.getStartState(), solution]
        # print(start_node)
        stack.push(start_node)
    if closed is None:
        closed = [problem.getStartState()]
    cur_node = stack.pop()
    next_successor = problem.getSuccessors(cur_node[0])

    for successor in next_successor:
        # print(cur_node[0])
        if successor[0] not in closed:
            move = cur_node[1].copy()
            move.append(successor[1])
            # print("move:", move)
            new_node = [successor[0], move]
            stack.push(new_node)
            Closed = closed.copy()
            Closed.append(successor[0])
            if problem.isGoalState(new_node[0]):
                return new_node[1]
            if depth + 1 < limit:
                new_DFS = depthFirstSearchLimit(problem, Closed, stack, depth + 1, limit)
                if new_DFS != []:
                    return new_DFS
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    import copy
    open = util.PriorityQueue()
    starth = heuristic(problem.getStartState(), problem)
    startg = 0
    startTrack = [None]
    initCost = 0
    bestG = 0
    start_node = [problem.getStartState(), startTrack, initCost]
    open.push(start_node, startg + 2*starth)
    closed = []
    best_g = {problem.getStartState(): 100000}
    while not open.isEmpty():
        s = open.pop()
        if s[0] not in closed or s[2] < best_g.get(s[0]):
            closed.append(s[0])
            best_g[s[0]] = s[2]
            if problem.isGoalState(s[0]):
                return s[1][1:]
            for c in problem.getSuccessors(s[0]):
                ch = heuristic(c[0], problem)
                cg = s[2] + c[2]
                Closed = copy.deepcopy(s[1])
                Closed.append(c[1])
                cData = [c[0], Closed, cg]
                if c[0] not in closed:
                    best_g[c[0]] = cg
                if heuristic(c[0], problem)<10000:
                    open.push(cData, cg + 2*ch)

def iterativeDeepeningSearch(problem):
    for i in range(100000):
        IDS = depthFirstSearchLimit(problem, None, None, 0, i+1)
        if IDS != []:
            return IDS

def waStarSearch(problem, heuristic=nullHeuristic):
    import copy
    open = util.PriorityQueue()
    h_state = heuristic(problem.getStartState(), problem)
    g_state = 0
    init_cost = 0
    solution = [None]
    start_node = [problem.getStartState(), solution, init_cost]
    open.push(start_node, g_state + 2 * h_state)
    closed = []
    best_g = {problem.getStartState(): 1000}
    while not open.isEmpty():
        cur_node = open.pop()
        if cur_node[0] not in closed or cur_node[2] < best_g.get(cur_node[0]):
            closed.append(cur_node[0])
            best_g[cur_node[0]] = cur_node[2]
            if problem.isGoalState(cur_node[0]):
                return cur_node[1][1:]
            for successors in problem.getSuccessors(cur_node[0]):
                cur_heur = heuristic(successors[0], problem)
                cur_g = cur_node[2] + successors[2]
                Closed = copy.deepcopy(cur_node[1])
                Closed.append(successors[1])
                New_node = [successors[0], Closed, cur_g]
                if successors[0] not in closed:
                    best_g[successors[0]] = cur_g
                if heuristic(successors[0], problem) < 1000:
                    open.push(New_node, cur_g + 2 * cur_heur)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch
dfsl = depthFirstSearchLimit
