from time import time
import argparse
import numpy as np
from numba import njit
import text_sudoku_dataset 

@njit(fastmath=True)
def calc_q(sudoku):
    
    '''
    Parameters
        sudoku: ndarray
            arrays of sudoku values
    Returns
        Q: ndarray
            updated unary penalties
    updates Q
    '''
    height,width = sudoku.shape
    n_labels = 9
    Q = np.zeros((height,width,n_labels),dtype=np.int32)
    

    for i in range(height):
        for j in range(width):
            if sudoku[i,j] != 0:
                # 1 if sudoku filled, 0 else
                Q[i,j,sudoku[i,j]-1] = 1
            else:
                Q[i,j,:] = 1
    return Q


def get_neighbours(i,j,indices_arr):

    '''
    Parameters
        i: int
            position along axis X
        j: int 
            position along axis Y
    Returns
        neighbours: ndarray
            indices of neighbours
    gives neighbours of sudoku[i,j] value
    '''
    # form list of neighbours to the current sudoku[i,j]
    horizontal = indices_arr[i,:]
    vertical = indices_arr[:,j]
    # form 3x3 cube based on current coordinate
    cube = indices_arr[(i//3)*3:(3+(i//3)*3),(j//3)*3:(3+(j//3)*3)].reshape(-1,2)
    
    neighbours = np.vstack((horizontal,vertical,cube))
    neighbours = np.unique(neighbours,axis=0).tolist()
    neighbours.remove([i,j])
    return neighbours




@njit(fastmath=True)
def calc_g(sudoku,neighbours_array):
    '''
    Parameters
        sudoku: ndarray
                arrays of sudoku values
        neighbours_array: ndarray
            array of neighbour indices to  sudoku value
    Returns
        G: ndarray
            binary penalties
    updates G
    '''
    height,width = sudoku.shape
    n_labels = 9
    n_neighbours = 20
    G = np.zeros((height,width,n_neighbours,n_labels,n_labels),dtype=np.int32)

    for i in range(height):
        for j in range(width):
            neighbours = neighbours_array[i,j,:,:]
            sudoku_value = sudoku[i,j]-1
            for n,[n_i,n_j] in enumerate(neighbours):
                neighbour_value = sudoku[n_i,n_j] - 1
                # 1 if sudoku and neigbours filled with some value
                if sudoku[i,j] != 0:
                    if sudoku[n_i,n_j] != 0 and sudoku_value != neighbour_value:
                        G[i,j,n,sudoku_value,neighbour_value] = 1
                    else:
                        # fill with ones other values,except outgoing edge
                        G[i,j,n,sudoku[i,j]-1,:] = 1
                        G[i,j,n,sudoku[i,j]-1,sudoku[i,j]-1] = 0
                else:
                    if sudoku[n_i,n_j] != 0:
                        G[i,j,n,:,neighbour_value] = 1
                        G[i,j,n,neighbour_value,neighbour_value] = 0
                    else:
                        G[i,j,n,:,:] = 1
    return G


@njit(fastmath=True)
def check_solution(Q_prev,G_prev,neighbours_array):

    '''
    Parameters
        Q_prev: ndarray
            arrays of unary penalties
        G_prev: ndarray
            array of binary penalties
        neighbours_array: ndarray
            array of neighbour indices to  sudoku value
    Returns
        True if solvable
        False if not
    check if there is solution based on Q and G
    '''
    Q_next = np.copy(Q_prev)
    G_next = np.copy(G_prev)
    height,width,n_neighbours,n_labels,_ = G_next.shape
    while True:
        for i in range(height):
            for j in range(width):
                neighbours = neighbours_array[i,j,:,:]
                for k in range(n_labels):
                    # check all neighbours according to formula 
                    for n,[n_i,n_j] in enumerate(neighbours):
                        binary = (G_prev[i,j,n,k,:] * Q_prev[n_i,n_j,:]).any()
                        # if encounter first 0 - everything is 0 (current Q and G)
                        if binary == 0:
                            Q_next[i,j,k] = 0
                            G_next[i,j,:,k,:] = 0
                            break
                        else:
                            G_next[i,j,n,k,:] = G_prev[i,j,n,k,:]  * Q_next[n_i,n_j,:]
        
        # check stop condition
        if (G_next == G_prev).all() and (Q_next == Q_prev).all():
            if np.prod(G_next == 0):
                return False
            else:
                return True 
        else:
            # update current Q,G and go to the next iteration
            Q_prev = np.copy(Q_next)
            G_prev = np.copy(G_next)


@njit(fastmath=True,nogil=True)
def solve_sudoku(sudoku,neighbours_array):

    '''
    Parameters
        sudoku: ndarray
                arrays of sudoku values
        neighbours_array: ndarray
            array of neighbour indices to  sudoku value
    Returns
        solved sudoku array or exception 'Refuse to recognize'

    solves sudoku
    '''
    height,width = sudoku.shape
    n_labels = 9
    # temp sudoku for filling with current values of K
    temp_sudoku = sudoku.copy()
    for i in range(height):
        for j in range(width):
            if temp_sudoku[i,j] == 0:
                for k in range(n_labels):
                    # fill temp sudoko with with current label 
                    temp_sudoku[i,j] = k + 1
                    Q,G = calc_q(temp_sudoku), calc_g(temp_sudoku,neighbours_array)
                    gamma = check_solution(Q,G,neighbours_array)
                    # if solvable, update solved sudoku
                    if gamma:
                        sudoku[i,j] = k+1
                        break
                    # if label not found - refuse to recognize
                    if k == n_labels - 1 and gamma == 0:
                        raise Exception('Refuse to recognize')
    return sudoku




def solver(sudoku):
    
    '''
    Parameters
        sudoku: ndarray
                arrays of sudoku value
    Returns
        solved sudoku array or exception 'Refuse to recognize'

    solves sudoku
    '''
    height,width = sudoku.shape
    n_neighbours = 20
    # array of possible indices for height x width array
    indices_arr = np.indices((height,width)).transpose(1,2,0)

    # all neighbours in one array to optimize computation
    neighbours_array = np.zeros((height,width,n_neighbours,2), dtype= np.int)
    for i in range(9):
        for j in range(9):
            # fill up array
            neighbours_array[i,j,:,:] = get_neighbours(i,j,indices_arr)

    solved = solve_sudoku(sudoku,neighbours_array)
    return solved







def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("sudoku_number", type=int, help="number of sudoku in dataset")
    args = parser.parse_args()

    sudoku,true_solved = text_sudoku_dataset.sudoku_dataset[args.sudoku_number]
    print('sudoku')
    print('*'*50)
    print(sudoku)
    solved = solver(sudoku)
    print('solution')
    print('*'*50)
    print(solved)


if __name__ == "__main__":
    main()


