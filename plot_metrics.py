from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import analytical as an
from simulated_metrics import customers_dist, wait_dist
from IPython.display import display
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulations import MD1Simulation, MM1Simulation

def _plot(simulated, CI, analytical, figsize = (10,5), **kwargs ):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(simulated, label = kwargs['simulated_label'])
    x = simulated.index
    y1 = CI.apply(lambda x: x[0])
    y2 = CI.apply(lambda x: x[1])
    
    ax.fill_between(x,  y1, y2, alpha = .5, label = 'confidence interval')
    
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

def _plot_pdf(simulated_distributions, analytical_pdf , figsize = (10,5), **kwargs):
    _kwargs = {
        'simulated_label' : 'simulated pdf',
        'analytical_label' : 'analytical pdf',
        'x_label' : 'queue_size',
        'y_label' : 'pdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['pdf'], simulated_distributions['pdf_CI'], analytical_pdf, figsize= figsize, **kwargs)

    
def _plot_cdf(simulated_distributions, analytical_cdf, figsize = (10,5), **kwargs):
    _kwargs = {
        'simulated_label' : 'simulated cdf',
        'analytical_label' : 'analytical cdf',
        'x_label' : 'queue_size',
         'y_label' : 'cdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['cdf'],simulated_distributions['cdf_CI'], analytical_cdf, figsize = figsize,**kwargs)

def plot_mm1_customers_dist(simulation_obj : MM1Simulation, figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)

    Q = an.mm1_markov_chain(simulation_obj.lamda, simulation_obj.mu, capacity = dists.index.max() + 1)
    pi = an.ctmc_stationary_distribution(Q)

    pdf = pi[:dists.index.max() + 1]
    cdf = pdf.cumsum()
    
    
    _plot_pdf(dists, pdf, figsize= figsize, **kwargs)
    _plot_cdf(dists, cdf, figsize= figsize, **kwargs)

def plot_md1_customers_dist(simulation_obj : MD1Simulation, figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)

    P = an.md1_markov_chain(simulation_obj.lamda, simulation_obj.mu, capacity = dists.index.max() + 1)
    pi = an.dtmc_stationary_distribution(P)
    pdf = pi[:dists.index.max() + 1]
    cdf = pi.cumsum()

    _plot_pdf(dists, pdf, figsize= figsize, **kwargs)
    _plot_cdf(dists, cdf, figsize= figsize, **kwargs)

def plot_mm1_wait_dist(simulation_obj : MM1Simulation, figsize = (10,5), **kwargs):
    pass

def plot_md1_wait_dist(simulation_obj : 'MD1Simulation', figsize = (10,5), **kwargs):
    pass