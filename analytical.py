import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm
from scipy.stats import poisson

def mm1_markov_chain(lamda, mu, capacity= 100):
    N = capacity
    M = np.zeros(shape = (N,N))
    M[0,1] = lamda
    M[0,0] = - lamda
    M[N -1, N-2 ] = mu
    M[N -1,N -1 ] = -mu
    for i in range(1, N - 1):
        M[i,i] = -(lamda + mu)
        M[i, i + 1] = lamda
        M[i, i -1] = mu
    return M

def mmk_markov_chain(lamda, mu, k, capacity = 100):
    N = capacity
    M = np.zeros(shape = (N,N))
    M[0,1] = lamda
    M[0,0] = - lamda
    M[N -1, N-2 ] = k * mu
    M[N -1,N -1 ] = - M[N -1, N-2 ]
    for i in range(1, N - 1):
        _mu = min( i , k) * mu
        M[i,i] = -(lamda + _mu)
        M[i, i + 1] = lamda
        M[i, i -1] = _mu
    return M
    
def mm1_markov_chain(lamda, mu, capacity= 100):
    N = capacity
    M = np.zeros(shape = (N,N))
    M[0,1] = lamda
    M[0,0] = - lamda
    M[N -1, N-2 ] = mu
    M[N -1,N -1 ] = -mu
    for i in range(1, N - 1):
        M[i,i] = -(lamda + mu)
        M[i, i + 1] = lamda
        M[i, i -1] = mu
    return M


def md1_markov_chain(lamda, mu, capacity = 100):
    rho = lamda/mu
    def alpha(k, rho):
        return poisson.pmf(k, rho)
    M = np.zeros(shape = (capacity,capacity))
    for i in range(capacity):
        lower = max(0,i - 1)
        for j in range(lower, capacity):
            M[i,j] = alpha(j - lower, rho)

    
    M[:, -1] = 1 - M.sum(axis = 1) + M[:, -1]
    return M

def ctmc_stationary_distribution(Q):
    # if 'capacity' in kwargs:
    #     Q = mm1_markov_chain(lamda, mu, capacity = kwargs['capacity'])
    # else:
    #     Q = mm1_markov_chain(lamda, mu)
    pi = expm(Q* 100000)

    if( np.isclose(pi, pi[0]).all() ):
        return pi[0]
    else:
        return None 

def dtmc_stationary_distribution(P):
    # if 'capacity' in kwargs:
    #     P = md1_markov_chain(lamda, mu, capacity = kwargs['capacity'])
    # else:
    #     P = md1_markov_chain(lamda, mu)
    pi = matrix_power(P, 10000)
    if( np.isclose(pi, pi[0]).all() ):
        return pi[0]
    else:
        return None 
        
def expected_value(pdf):
    return sum( [i * pdf[i] for i in range(len(pdf)) ] )



