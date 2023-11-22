# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.


        Important things to consider to evaluate game board:
        how much food is left
        how far away you are from food
        how far away you are from ghosts
        if all food is eaten you win
        if ghost is at your position you lose


        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        minFoodDist = 1e9
        if not newFood:
            minFoodDist = 0

        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    minFoodDist = min(minFoodDist, manhattanDistance(newPos, (x, y)))

        minGhostDist = 1e9
        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                minGhostDist = min(minGhostDist, manhattanDistance(newPos, ghost.getPosition()))

        return successorGameState.getScore() + (1/minFoodDist) - (1/minGhostDist)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.MiniMax(gameState, self.depth, 0)[0]

    def MiniMax(self, state, depth, agentNum):
        if depth == 0 or state.isWin() or state.isLose():
            return Directions.STOP, self.evaluationFunction(state)
        elif agentNum == 0:
            return self.Maxi(state, depth, agentNum)
        else:
            return self.Mini(state, depth, agentNum)

    def Mini(self, state, depth, agentNum):
        moves = state.getLegalActions(agentNum)

        newAgent = (agentNum + 1) % state.getNumAgents()

        if newAgent == 0:
            newDepth = depth - 1
        else:
            newDepth = depth

        minScore = 1e9
        minAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.MiniMax(newState, newDepth, newAgent)

            if newScore < minScore:
                minAction = move

            minScore = min(minScore, newScore)

        return minAction, minScore

    def Maxi(self, state, depth, agentNum):
        moves = state.getLegalActions(agentNum)

        newAgent = (agentNum + 1) % state.getNumAgents()

        newDepth = depth

        maxScore = -1e9
        maxAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.MiniMax(newState, newDepth, newAgent)

            if newScore > maxScore:
                maxAction = move

            maxScore = max(maxScore, newScore)
        return maxAction, maxScore

        #util.raiseNotDefined()




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.MiniMaxAB(gameState, self.depth, 0, -1e9, 1e9)[0]
        util.raiseNotDefined()

    def MiniMaxAB(self, state, depth, agentNum, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return Directions.STOP, self.evaluationFunction(state)
        elif agentNum == 0:
            return self.MaxiAB(state, depth, agentNum, alpha, beta)
        else:
            return self.MiniAB(state, depth, agentNum, alpha, beta)

    def MiniAB(self, state, depth, agentNum, alpha, beta):
        moves = state.getLegalActions(agentNum)

        newAgent = (agentNum + 1) % state.getNumAgents()

        if newAgent == 0:
            newDepth = depth - 1
        else:
            newDepth = depth

        minScore = 1e9
        minAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.MiniMaxAB(newState, newDepth, newAgent, alpha, beta)

            if newScore < minScore:
                minAction = move

            minScore = min(minScore, newScore)

            if minScore < alpha:
                return minAction, minScore
            beta = min(beta, minScore)

        return minAction, minScore

    def MaxiAB(self, state, depth, agentNum, alpha, beta):
        moves = state.getLegalActions(agentNum)
        newAgent = (agentNum + 1) % state.getNumAgents()

        newDepth = depth

        maxScore = -1e9
        maxAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.MiniMaxAB(newState, newDepth, newAgent, alpha, beta)

            if newScore > maxScore:
                maxAction = move

            maxScore = max(maxScore, newScore)

            if maxScore > beta:
                return maxAction, maxScore
            alpha = max(alpha, maxScore)

        return maxAction, maxScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):


        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectimaxSearch(gameState, self.depth, 0)[0]

    def ExpectimaxSearch(self, state, depth, agentNum):
        if depth == 0 or state.isWin() or state.isLose():
            return Directions.STOP, self.evaluationFunction(state)
        elif agentNum == 0:
            return self.Maxi(state, depth, agentNum)
        else:
            return self.Expectimax(state, depth, agentNum)

    def Maxi(self, state, depth, agentNum):
        moves = state.getLegalActions(agentNum)

        newAgent = (agentNum + 1) % state.getNumAgents()

        newDepth = depth

        maxScore = -1e9
        maxAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.ExpectimaxSearch(newState, newDepth, newAgent)

            if newScore > maxScore:
                maxScore = newScore
                maxAction = move

        return maxAction, maxScore

    def Expectimax(self, state, depth, agentNum):
        moves = state.getLegalActions(agentNum)

        newAgent = (agentNum + 1) % state.getNumAgents()

        if newAgent == 0:
            newDepth = depth - 1
        else:
            newDepth = depth

        ExpectedScore = 0
        ExpectedAction = Directions.STOP

        for move in moves:
            newState = state.generateSuccessor(agentNum, move)
            retAction, newScore = self.ExpectimaxSearch(newState, newDepth, newAgent)
            ExpectedScore += newScore/len(moves)

        return ExpectedAction, ExpectedScore



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    "*** YOUR CODE HERE ***"
    minFoodDist = 100000000000
    if not Food:
        minFoodDist = 0

    for x in range(Food.width):
        for y in range(Food.height):
            if Food[x][y]:
                minFoodDist = min(minFoodDist, manhattanDistance(Pos, (x, y)))

    minGhostDist = 10000000000
    for ghost in GhostStates:
        if ghost.scaredTimer > 0:
            minGhostDist = min(minGhostDist, manhattanDistance(Pos, ghost.getPosition()))

    return (1/minFoodDist) - (1/minGhostDist) + currentGameState.getScore()

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
