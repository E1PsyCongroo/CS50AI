"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    X is the first, O is the next.
    if the board is the end of the game, any return value is ok.
    """
    count_X = 0
    count_O = 0
    for i in board:
        for j in i:
            if j == X:
                count_X += 1
            elif j == O:
                count_O += 1
    if count_X <= count_O:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    result = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                result.add((i, j))
    return result

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] is not EMPTY:
        raise Exception("action is not a valid action for the board")
    new_board = [[j for j in i] for i in board]
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] == board[1][1] == board[2][2] or board[2][0] == board[1][1] == board[0][2]:
        return board[1][1]
    for i in range(3):
        start = board[i][i]
        count_row = 0
        count_col = 0
        for j in range(3):
            if board[i][j] == start:
                count_row += 1
            if board[j][i] == start:
                count_col += 1
        if count_row == 3 or count_col ==3:
            return start
    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    theWinner = winner(board)
    if theWinner == X:
        return 1
    elif theWinner == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    MAX = X win
    MIN = O win
    """
    def max_action(board):
        max = [-1, None]
        if terminal(board):
            return [utility(board), None]
        for action in actions(board):
            min_value = min_action(result(board, action))[0]
            if min_value >= max[0]:
                max = [min_value, action]
        return max

    def min_action(board):
        min = [1, None]
        if terminal(board):
            return [utility(board), None]
        for action in actions(board):
            max_value = max_action(result(board, action))[0]
            if max_value <= min[0]:
                min = [max_value, action]
        return min

    def max_action_alpha_beta_puring(board):
        max = [-1, None]
        if terminal(board):
            return [utility(board), None]
        for action in actions(board):
            min_value = min_action(result(board, action))[0]
            if min_value >= max[0]:
                max = [min_value, action]
            if max[0] == 1:  # Alpha-Beta-Puring 剪枝
                return max
        return max

    def min_action_alpha_beta_puring(board):
        min = [1, None]
        if terminal(board):
            return [utility(board), None]
        for action in actions(board):
            max_value = max_action(result(board, action))[0]
            if max_value <= min[0]:
                min = [max_value, action]
            if min[0] == -1:  # Alpha-Beta-Puring 剪枝
                return min
        return min

    turn = player(board)
    """
    非剪枝版本
    if turn == X:
        return max_action(board)[1]
    elif turn == O:
        return min_action(board)[1]
    """
    #剪枝版本
    if turn == X:
        return max_action_alpha_beta_puring(board)[1]
    elif turn == O:
        return min_action_alpha_beta_puring(board)[1]
