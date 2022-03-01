import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm

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

def stationary_distribution(lamda, mu, **kwargs):
    if 'capacity' in kwargs:
        Q = mm1_markov_chain(lamda, mu, capacity = kwargs['capacity'])
    else:
        Q = mm1_markov_chain(lamda, mu)

    stationary_chain = expm(Q* 1000)

    if( np.isclose(stationary_chain, stationary_chain[0]).all() ):
        return stationary_chain[0]
    else:
        return None 

def expected_value(pdf):
    return sum( [i * pdf[i] for i in range(len(pdf)) ] )
