from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import analytical as an
from simulated_metrics import customers_dist, wait_dist
from IPython.display import display
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulations import MD1Simulation, MM1Simulation



def _plot(simulated, CI, analytical, figsize = (10,5), plot_analytical = True, **kwargs ):
    fig, ax = plt.subplots(figsize = figsize)

    ax.plot(simulated, label = kwargs['simulated_label'])

    x = simulated.index
    y1 = CI.apply(lambda x: x[0])
    y2 = CI.apply(lambda x: x[1])
    ax.fill_between(x,  y1, y2, alpha = .5, label = 'confidence interval')

    def _plot_analytical():
        ax.plot(analytical, color = 'red', linestyle = '--', dashes = (5,10), label = kwargs['analytical_label'])
    
    if(plot_analytical):
        _plot_analytical()

    
    if('xticks' in kwargs.keys()):
        ax.set_xticks(kwargs['xticks'])
    if('xlabels' in kwargs.keys()):
        ax.set_xticklabels(kwargs['xlabels'], rotation = 45)
    else:
        ax.set_xticklabels(ax.get_xticks(), rotation = 45)

    ax.legend()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])

    #y_ticks = np.linspace(start = min_y, stop = max_y, num = 10)
    #ax.set_yticks(y_ticks)
    #ax.set_xlim(left = 0)

    #plota a linha y = 0
    #plt.axhline(y=0, color='black', linewidth = 1, linestyle='-')

    fig.patch.set_facecolor('white')
    plt.grid( linestyle = '--')
    plt.show()

def _plot_pdf(simulated_distributions, analytical_pdf , plot_analytical = True, figsize = (10,5), **kwargs):
    _kwargs = {
        'simulated_label' : 'simulated pdf',
        'analytical_label' : 'analytical pdf',
        'y_label' : 'pdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['pdf'], simulated_distributions['pdf_CI'], analytical_pdf, plot_analytical= plot_analytical, figsize= figsize, **kwargs)

    
def _plot_cdf(simulated_distributions, analytical_cdf,  plot_analytical = True, figsize = (10,5), **kwargs):
    _kwargs = {
        'simulated_label' : 'simulated cdf',
        'analytical_label' : 'analytical cdf',
        'y_label' : 'cdf'
    }
    kwargs.update(_kwargs)
    _plot(simulated_distributions['cdf'],simulated_distributions['cdf_CI'], analytical_cdf, plot_analytical= plot_analytical, figsize = figsize,**kwargs)

#gera ticks do eixo x do plot de clientes
def _customers_ticks(data, max_ticks = -1):
    x = data.index
    total_ticks = max_ticks if(max_ticks != -1) else len(x)

    xticks_dist = len(x)/total_ticks
    xticks = np.linspace(0, xticks_dist * total_ticks , total_ticks + 1, dtype = np.int64 )
    return xticks

#gera ticks do eixo x do plot de esperas
def _waits_ticks(data):
    xticks = [x[0] for x in data.index]
    xlabels = [f'{x[0]:.2f}' for x in data.index]
    return xticks, xlabels

def plot_mm1_customers_dist(simulation_obj : MM1Simulation, figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)

    should_plot_analytical = simulation_obj.rho < 1
    Q = an.mm1_markov_chain(simulation_obj.lamda, simulation_obj.mu, capacity = dists.index.max() + 1)
    pi = an.ctmc_stationary_distribution(Q)

    pdf = pi[:dists.index.max() + 1]
    cdf = pdf.cumsum()

    max_ticks = kwargs['max_ticks'] if 'max_ticks' in kwargs.keys() else -1
    xticks = _customers_ticks(dists, max_ticks = max_ticks)

    _kwargs = {
        'x_label' : 'queue_size',
        'xticks' : xticks
    }
    
    kwargs.update(_kwargs)
    _plot_pdf(dists, pdf, figsize= figsize, plot_analytical= should_plot_analytical, **kwargs)
    _plot_cdf(dists, cdf, figsize= figsize, plot_analytical= should_plot_analytical, **kwargs)

def plot_md1_customers_dist(simulation_obj : MD1Simulation, figsize = (10,5), **kwargs):
    dists = customers_dist(simulation_obj)
    display(dists)

    should_plot_analytical = simulation_obj.rho < 1

    max_ticks = kwargs['max_ticks'] if 'max_ticks' in kwargs.keys() else -1
    xticks = _customers_ticks(dists, max_ticks=max_ticks)

    _kwargs = {
        'x_label' : 'queue_size',
        'xticks' : xticks
    }
    kwargs.update(_kwargs)
    if(should_plot_analytical):
        P = an.md1_markov_chain(simulation_obj.lamda, simulation_obj.mu, capacity = dists.index.max() + 1)
        pi = an.dtmc_stationary_distribution(P)
        pdf = pi[:dists.index.max() + 1]
        cdf = pi.cumsum()
    else:
        pdf = None,
        cdf = None

    _plot_pdf(dists, pdf, figsize= figsize, plot_analytical =should_plot_analytical,  **kwargs)
    _plot_cdf(dists, cdf, figsize= figsize, plot_analytical =should_plot_analytical, **kwargs)

def plot_mm1_wait_dist(simulation_obj : MM1Simulation, figsize = (10,5), **kwargs):
    waits = wait_dist(simulation_obj)
    display(waits)
    xticks, xlabels = _waits_ticks(waits)
    _kwargs = {
        'x_label' : 'mean_wait',
        'xticks' : xticks,
        'xlabels' : xlabels
    }
    kwargs.update(_kwargs)
    waits.index = [x[0] for x in waits.index]
    #_plot_pdf(waits, None, plot_analytical = False, figsize= figsize, **kwargs)
    _plot_cdf(waits, None, plot_analytical = False, figsize= figsize, **kwargs)

    return waits

def plot_md1_wait_dist(simulation_obj : MD1Simulation, figsize = (10,5), **kwargs):
    waits = wait_dist(simulation_obj)
    display(waits)
    xticks, xlabels = _waits_ticks(waits)
    _kwargs = {
        'x_label' : 'mean_wait',
        'xticks' : xticks,
        'xlabels' : xlabels
    }
    kwargs.update(_kwargs)
    waits.index = [x[0] for x in waits.index]
    #_plot_pdf(waits, None, plot_analytical = False, figsize= figsize, **kwargs)
    _plot_cdf(waits, None, plot_analytical = False, figsize= figsize, **kwargs)
    return waits

def plot_average_metrics(simulations_obj:  list(MD1Simulation), figsize = (10,5), **kwargs):
    simulations = [obj.data for obj in simulations_obj]

