import numpy as np
import matplotlib.pyplot as plt
import analytical as an
from simulated_metrics import customers_dist
from IPython.display import display

def _plot(simulated, analytical, figsize = (10,5), **kwargs ):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(simulated, label = kwargs['simulated_label'])
    x = simulated.index
    # y1 = CI.apply(lambda x: x[0])
    # y2 = CI.apply(lambda x: x[1])
    
    # ax.fill_between(x,  y1, y2, alpha = .5, label = 'confidence interval')
    
    ax.plot(analytical, color = 'red', linestyle = '--', dashes = (5,10), label = kwargs['analytical_label'])
    ax.legend()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    
    ticks_dist = kwargs['xticks_dist'] if('xticks_dist' in kwargs.keys()) else 1
    

    total_ticks = x.max()/ticks_dist
    total_ticks = int(total_ticks) + 1
    ax.set_xticks( np.linspace(0, ticks_dist * total_ticks , total_ticks + 1, dtype = int ))
    ax.set_xticklabels(ax.get_xticks(), rotation = 90)

    
    max_y = max( simulated.max(), analytical.max() ) + max(simulated.std(), analytical.std() )
    ax.set_yticks(np.linspace(0, max_y , 11 ))
    ax.set_xlim(left = 0)

    fig.patch.set_facecolor('white')
    plt.grid( linestyle = '--')
    plt.show()

def _plot_pdf(simulated_distributions, lamda, mu, figsize = (10,5), kind = 'm', **kwargs):
    assert kind in ['m', 'd']
    if kind == 'm':
        Q = an.mm1_markov_chain(lamda, mu, capacity = simulated_distributions.index.max() + 1)
        pi = an.ctmc_stationary_distribution(Q)
        pi = pi[:simulated_distributions.index.max() + 1]
    else:
        pi = an.md1_markov_chain(lamda, mu, capacity = simulated_distributions.index.max() + 1)
        pi = np.linalg.matrix_power(pi, 1000)[0]
    _kwargs = {
        'simulated_label' : 'simulated pdf',
        'analytical_label' : 'analytical pdf',
        'x_label' : 'queue_size',
        'y_label' : 'pdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['pdf'], pi, figsize= figsize, **kwargs)

    
def _plot_cdf(simulated_distributions, lamda, mu,figsize = (10,5),  kind = 'm',  **kwargs):
    assert kind in ['m', 'd']
    if kind == 'm':
        Q = an.mm1_markov_chain(lamda, mu, capacity = simulated_distributions.index.max() + 1)
        pi = an.ctmc_stationary_distribution(Q)
        pi = pi.cumsum()
        pi = pi[:simulated_distributions.index.max() + 1]
    else:
        pi = an.md1_markov_chain(lamda, mu, capacity = simulated_distributions.index.max() + 1)
        pi = np.linalg.matrix_power(pi, 1000)[0]
        pi = pi.cumsum()

    _kwargs = {
        'simulated_label' : 'simulated cdf',
        'analytical_label' : 'analytical cdf',
        'x_label' : 'queue_size',
         'y_label' : 'cdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['cdf'], pi, figsize = figsize,**kwargs)
    
def plot_mm1_customers_dist(simulation_obj : 'MM1Simulation', figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)
    _plot_pdf(dists, simulation_obj.lamda, simulation_obj.mu, figsize= figsize, kind='m', **kwargs)
    _plot_cdf(dists, simulation_obj.lamda, simulation_obj.mu, figsize= figsize, kind='m', **kwargs)

def plot_md1_customers_dist(simulation_obj : 'MD1Simulation', figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)
    _plot_pdf(dists, simulation_obj.lamda, simulation_obj.mu, figsize= figsize, kind='d', **kwargs)
    _plot_cdf(dists, simulation_obj.lamda, simulation_obj.mu, figsize= figsize, kind='d', **kwargs)

