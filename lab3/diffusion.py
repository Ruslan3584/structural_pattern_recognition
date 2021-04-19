import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numba import njit

@njit
def get_neighbours(height,width,i,j):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        i: int
            number of row
        j: int
            number of column
    Returns
        N: tuple
            neighbours coordinates, neighbours indices, inverse neighbours indices
    calculates neighbours in 4-neighbours system (and inverse indices)
    
    
             (i-1,j-1)  (i-1,j)  (i-1,j+1) 
                      \    |    /
                       \   |   /
                        \  |  /
                (i,j-1)--(i,j)--(i,j+1)
                        /  |  \
                       /   |   \
                      /    |    \
             (i+1,j-1)  (i+1,j)  (i+1,j+1)
    '''

    # i,j - position of pixel
    # [Left, Right, Up, Down] - order of possible neighbours
    # array of neighbour indices

    nbs = []
    # neighbour indices
    nbs_indices = []
    # inverse neighbour indices
    inv_nbs_indices = []
    # Top Left
    if 0<=i-1<height-1 and 0<=j-1<width-1:
        nbs.append([i-1,j-1])
        inv_nbs_indices.append(7)
        nbs_indices.append(0)
        
    # Top Up
    if 0<=i-1<height-1 and 0<=j<=width-1:
        nbs.append([i-1,j])
        inv_nbs_indices.append(6)
        nbs_indices.append(1)        

    # Top Right
    if 0<=i-1<height-1 and 0<j+1<=width-1:
        nbs.append([i-1,j+1])
        inv_nbs_indices.append(5)
        nbs_indices.append(2)
    # Left
    if 0<=i<=height-1 and 0<=j-1<width-1:
        nbs.append([i,j-1])
        inv_nbs_indices.append(4)
        nbs_indices.append(3)
    # Right
    if 0<=i<=height-1 and 0<j+1<=width-1:
        nbs.append([i,j+1])
        inv_nbs_indices.append(3)
        nbs_indices.append(4)
    # Bottom Left
    if 0<i+1<=height-1 and 0<=j-1<width-1:
        nbs.append([i+1,j-1])
        inv_nbs_indices.append(2)
        nbs_indices.append(5)
    # Bottom Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        nbs.append([i+1,j])
        inv_nbs_indices.append(1)
        nbs_indices.append(6)
    # Bottom Right
    if 0<i+1<=height-1 and 0<j+1<=width-1:
        nbs.append([i+1,j+1])
        inv_nbs_indices.append(0)
        nbs_indices.append(7)
    
    N = (nbs, inv_nbs_indices, nbs_indices)
    return N


@njit(fastmath=True, cache=True)
def diff_iter(height: int, width: int, fi: ndarray, g:ndarray, K:ndarray, Q:ndarray) -> ndarray:

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        fi: ndarray
            array of potentials
        g: ndarray
            array of binary penalties
        K: ndarray
            label set
        Q: ndarray
            array of unary penalties
    Returns
        fi: ndarray
            updated array of potentials

    calculates one iteration of diffusion 
    '''
    # for each pixel of input image
    for i in range(height):
        for j in range(width):
            # defining neighbours pixel (i,j)
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            len_neighbours = len(nbs)
            k_asterisk = np.full(len_neighbours, -1)
            fi_list = np.full(len_neighbours, 0)
            C_t = np.full(len_neighbours, 0)
            # for each label in label set
            for k in K:
                # for each neighbour of pixel (i,j)
                for n,[n_i,n_j] in enumerate(nbs):
                    # k*(t`)
                    k_asterisk[n] = np.argmax(g[i,j,n,k,:] - fi[n_i,n_j,inv_nbs_indices[n],:])
                    # fi(t',t)(k*)
                    fi_list[n] = fi[n_i,n_j,inv_nbs_indices[n],k_asterisk[n] ]
                    if g[i,j,n,k,k_asterisk[n]] == -np.inf:
                        C_t[n] = -np.inf
                    else:
                        C_t[n] = g[i,j,n,k,k_asterisk[n]] - fi_list[n]
                #print(C_t)
                C_t_sum = (np.sum(C_t) + Q[i,j,k])/len_neighbours
                # updating potentials
                for n in range(len_neighbours):
                    fi[i,j,nbs_indices[n],k] = C_t[n] - C_t_sum    
    return fi

def diffusion(height,width,K,Q,g, n_iter):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        K: ndarray
            label set
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        n_iter: int
            number of iteration
    Returns
        fi: ndarray
            updated array of potentials after n iterations

    n iterations of diffusion
    examples:
    >>> diffusion('height','width','K','Q','g', 0)
    Traceback (most recent call last):
    ...
    Exception: number of iterations is less or equal to zero
    '''

    if n_iter <= 0:
        raise Exception('number of iterations is less or equal to zero')

    n_labels = len(K)
    # 8 system neighbourhood
    n_neighbors = 8
    # initialize potentials as zeros
    fi = np.zeros((height,width,n_neighbors,n_labels))
    for i in range(n_iter):
        fi = diff_iter(height,width,fi,g,K,Q)
    return fi

def get_labelling(height, width, g, fi):

    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        g: ndarray
            array of binary penalties
        c: ndarray
            color mapping
        fi: ndarray
            updated array of potentials
    Returns
        labelling: ndarray
            optimal labels after diffusion algorithm

    labelling restoration after diffusion
    '''
    c = np.array([[0,0,255],[255,0,0],[0,255,0]])
    labelling = np.empty((height,width,len(c[0])),dtype = int)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            # take any neighbour
            n_i, n_j = nbs[0]
            # calculating reparametrized binary penalties
            g_reparametrized = g[i,j,nbs_indices[0],:] - fi[i,j,nbs_indices[0],:] - fi[n_i,n_j,inv_nbs_indices[0],:]
            # g - is supermodular so take the highest possible maximum edge between nodes t, t'
            labelling[i,j,:] = c[np.argmax(np.max(g_reparametrized,axis = 0))]
    return labelling

