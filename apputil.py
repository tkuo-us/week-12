import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    # your code here ...
    # updated_board = current_board

    # transform input to integer type for easier calculations
    board = current_board.astype(int)
    rows, cols = board.shape

    # create an empty board for the updated state
    updated_board = np.zeros((rows, cols), dtype=int)

    # calculate the next state for each cell
    for i in range(rows):
        for j in range(cols):
            # get 3x3 neighborhood
            r_min, r_max = max(i-1, 0), min(i+2, rows)
            c_min, c_max = max(j-1, 0), min(j+2, cols)
            neighbors = board[r_min:r_max, c_min:c_max].sum() - board[i, j]

            # according to the rules of Conway's Game of Life
            if board[i, j] == 1:    # Survival
                if neighbors == 2 or neighbors == 3:
                    updated_board[i, j] = 1
            else:  # Death
                if neighbors == 3:
                    updated_board[i, j] = 1  # becomes alive

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)


def recursive_game(step=0, max_steps=10, board=None):
    # first call: initialize board if not provided
    if board is None:
        board = np.random.randint(2, size=(10, 10))

    # base case: check if max steps reached
    if step >= max_steps:
        return board

    # update board
    from apputil import update_board
    new_board = update_board(board)

    # recursive case: call function again with incremented step
    return recursive_game(step + 1, max_steps, new_board)
