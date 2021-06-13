import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
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
            (i-1,j)
               |
    (i,j-1)--(i,j)--(i,j+1)
               |
            (i+1,j)
    examples:
    >>> get_neighbours(-1,1,1,1)
    Traceback (most recent call last):
    ...
    Exception: height or width is less than zero
    >>> get_neighbours(3,3,-1,-1)
    ([], [], [])
    >>> get_neighbours(3,3,0,0)
    ([[0, 1], [1, 0]], [0, 2], [1, 3])
    >>> get_neighbours(3,3,0,1)
    ([[0, 0], [0, 2], [1, 1]], [1, 0, 2], [0, 1, 3])
    >>> get_neighbours(3,3,1,1)
    ([[1, 0], [1, 2], [0, 1], [2, 1]], [1, 0, 3, 2], [0, 1, 2, 3])
    '''

    if width <= 0 or height <= 0:
        raise Exception('height or width is less than zero')

    # i,j - position of pixel
    # [Left, Right, Up, Down] - order of possible neighbours
    # array of neighbour indices
    nbs = [] 
    # neighbour indices
    nbs_indices = []
    # inverse neighbour indices
    inv_nbs_indices = []
    # Left
    if 0<=j-1<width-1 and 0<=i<=height-1:
        nbs.append([i,j-1])
        inv_nbs_indices.append(1)
        nbs_indices.append(0)
    # Right
    if 0<j+1<=width-1 and 0<=i<=height-1:
        nbs.append([i,j+1])
        inv_nbs_indices.append(0)
        nbs_indices.append(1)
    # Upper
    if 0<=i-1<height-1 and 0<=j<=width-1:
        nbs.append([i-1,j])
        inv_nbs_indices.append(3)
        nbs_indices.append(2)
    # Down
    if 0<i+1<=height-1 and 0<=j<=width-1:
        nbs.append([i+1,j])
        inv_nbs_indices.append(2)
        nbs_indices.append(3)
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
            fi_list = np.full(len_neighbours, np.nan)
            C_t = np.full(len_neighbours, np.nan)
            # for each label in label set
            for k in K:
                # for each neighbour of pixel (i,j)
                for n,[n_i,n_j] in enumerate(nbs):
                    # k*(t`)
                    k_asterisk[n] = np.argmax(g[i,j,nbs_indices[n],k,:] - fi[n_i,n_j,inv_nbs_indices[n],:])
                    # fi(t',t)(k*)
                    fi_list[n] = fi[n_i,n_j,inv_nbs_indices[n],k_asterisk[n] ]
                    if g[i,j,nbs_indices[n],k,k_asterisk[n]] == -np.inf:
                        C_t[n] = -np.inf
                    else:
                        C_t[n] = g[i,j,nbs_indices[n],k,k_asterisk[n]] - fi_list[n]
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
    n_neighbors = 4
    # initialize potentials as zeros
    fi = np.zeros((height,width,n_neighbors,n_labels))
    for i in range(n_iter):
        fi = diff_iter(height,width,fi,g,K,Q)
    return fi

def update_fi(height,width,fi,g,K,Q,n_iter):
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
        n_iter: int
            number of iteration
    Returns
        fi: ndarray
            updated array of potentials
    updates fi after n_iter iterations
    '''
    n_labels = len(K)
    n_neighbors = 4
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
        fi: ndarray
            updated array of potentials
    Returns
        labelling: ndarray
            optimal labels after diffusion algorithm

    labelling restoration after diffusion
    '''
    labelling = np.empty((height,width),dtype = int)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height, width,i,j)
            n_i, n_j = nbs[0]
            # calculating reparametrized binary penalties
            g_reparametrized = g[i,j,nbs_indices[0],...] - fi[i,j,nbs_indices[0],...] - fi[n_i,n_j,inv_nbs_indices[0],...]
            # g - is supermodular so take the highest possible maximum edge between nodes t, t'
            labelling[i,j] = np.argmax(np.max(g_reparametrized,axis = 0))
    return labelling


def get_gmm_params(img, n_rows):
    width = img.shape[1]
    class0 = GaussianMixture(n_components=1).fit(img.reshape(-1,3)[:(n_rows*width)] )
    class1 = GaussianMixture(n_components=1).fit(img.reshape(-1,3)[(n_rows*width):] )
    return class0.means_[0], class0.covariances_[0], class1.means_[0],  class1.covariances_[0]

def get_q(img, mean0, cov0, mean1, cov1):
    height, width = img.shape[:2]
    K = np.array([0,1])
    mask = multivariate_normal.pdf(img, mean0, cov0) > multivariate_normal.pdf(img, mean1, cov1)
    q = np.zeros((height,width, 2))
    q[mask] = (K != 1)
    q[~mask] = (K != 0)
    return -q

def get_g(height,width,n_labels,beta):
    g = np.zeros((height,width, 4, 2, 2))
    g[..., 0, 1] = -beta
    g[..., 1, 0] = -beta
    return g

@njit
def check_smooth_border(mask, height, width):
    for i in range(height-1):
        for j in range(width):
            if (not mask[i, j]) and mask[i + 1, j] and (mask[i+1:, j] == 0).any():
                    return False
            if (not mask[i, j]) and (not mask[i+1, j]):
                if  j != 0 and mask[i, j-1] and mask[i+1, j-1]:
                    return False
                if  j < width - 1 and mask[i, j+1] and mask[i+1, j+1]:
                    return False
    return True




@njit
def get_reparametrized_q(q, fi):
    height, width, n_labels = q.shape
    q_reparametrized = np.copy(q).astype(np.float64)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height,width,i,j)
            for k in range(n_labels):
                for n,[n_i,n_j] in enumerate(nbs):
                    q_reparametrized[i,j,k] += fi[i,j,nbs_indices[n],k]
    return q_reparametrized



@njit
def get_reparametrized_g(g,fi):
    height, width = fi.shape[:2]
    n_labels = g.shape[-1]
    g_reparametrized = np.full_like(g, -np.inf)
    eps_array = np.zeros_like(g)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height,width,i,j)
            for n,[n_i,n_j] in enumerate(nbs):
                # calculating reparametrized binary penalties
                for k in range(n_labels):
                    for k_prime in range(n_labels):
                        g_reparametrized[i,j,nbs_indices[n],k,k_prime] = g[i,j,nbs_indices[n],k,k_prime] - fi[i,j,nbs_indices[n],k]  - fi[n_i,n_j,inv_nbs_indices[n],k_prime]
                g_max = np.max(g_reparametrized[i,j,nbs_indices[n],...])
                eps_array[i,j,nbs_indices[n],:] = np.abs(g_max - g_reparametrized[i,j,nbs_indices[n],...])
                #print(np.unique(eps_array[i,j,nbs_indices[n],:])[-2])
    eps_unique = np.unique(eps_array)
    return g_reparametrized, eps_unique


@njit
def get_g_binary(g_reparametrized, epsilon):
    height, width = g_reparametrized.shape[:2]
    g_binary = np.full_like(g_reparametrized, 0)
    for i in range(height):
        for j in range(width):
            nbs, inv_nbs_indices, nbs_indices = get_neighbours(height,width,i,j)
            for n,[n_i,n_j] in enumerate(nbs):
                    g_max = np.max(g_reparametrized[i,j,nbs_indices[n],:])
                    g_binary[i,j,nbs_indices[n],:] = (np.abs(g_max - g_reparametrized[i,j,nbs_indices[n],:]) < epsilon)
    return g_binary


@njit
def get_q_binary(q_reparametrized, epsilon):
    height, width = q_reparametrized.shape[:2]
    q_binary = np.full_like(q_reparametrized, 0)
    for i in range(height):
        for j in range(width):
            q_max = np.max(q_reparametrized[i,j,:])
            q_binary[i,j,:] =  (np.abs(q_max - q_reparametrized[i,j,:]) < epsilon)
    return q_binary


@njit
def crossing_out(q, g):
    height, width = g.shape[:2]
    n_labels = g.shape[-1]
    change_indicator = True
    while(change_indicator):
        change_indicator = False
        for i in range(height):
            for j in range(width):
                nbs, inv_nbs_indices, nbs_indices = get_neighbours(height,width,i,j)
                for k in range(n_labels):
                    if q[i,j,k] == 0:
                        for n,[n_i,n_j] in enumerate(nbs):
                            if (g[i,j,nbs_indices[n],k,:] == 1).any():
                                g[i,j,nbs_indices[n],k,:] = 0
                                g[n_i,n_j,inv_nbs_indices[n],:,k] = 0
                                change_indicator = True
                    elif q[i,j,k] == 1:
                        exists = 0
                        edge_exists = False
                        for n,[n_i,n_j] in enumerate(nbs):
                            for k1 in range(n_labels):
                                if g[i,j,nbs_indices[n],k,k1] == 1 and q[n_i,n_j,k1] == 1:
                                    edge_exists = True
                                    break
                            if edge_exists == True:
                                exists += 1
                        if exists != len(nbs):
                            for n,[n_i,n_j] in enumerate(nbs):
                                if (g[i,j,nbs_indices[n],k,:]).any() == 1:
                                    g[i,j,nbs_indices[n],k,:] = 0
                                    g[n_i,n_j,inv_nbs_indices[n],:,k] = 0
                                    change_indicator = True
                                if q[i,j,k] == 1:
                                    q[i,j,k] = 0
                                    change_indicator = True
        if (q == 0).all() or (g == 0).all():
            print("ALL ZEROS")
            return 0
    return 1

@njit
def self_control(q_binary,g_binary):
    height,width,n_labels = q_binary.shape
    temp_result = np.ones((height,width))
    for i in range(height):
        for j in range(width):
            for k in range(n_labels):
                if q_binary[i,j,k] == 1:
                    q_binary[i,j,:] = 0
                    q_binary[i,j,k] = 1
                    flag = crossing_out(q_binary, g_binary)
                    if flag == 1:
                        temp_result[i,j] = k
                        break
                    if k == n_labels - 1 and flag == 0:
                        print('bad parameters')
    result = temp_result[1:,1:]
    return result




