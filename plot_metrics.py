import numpy as np
import matplotlib.pyplot as plt
import analytical as an


def plot(simulated, CI, analytical, figsize = (10,5), **kwargs ):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(simulated, label = kwargs['simulated_label'])
    y1 = CI.apply(lambda x: x[0])
    y2 = CI.apply(lambda x: x[1])
    x = simulated.index
    ax.fill_between(x,  y1, y2, alpha = .5, label = 'confidence interval')
    
    ax.plot(analytical, color = 'red', linestyle = '--', dashes = (5,10), label = kwargs['analytical_label'])
    ax.legend()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    
    if('xticks_dist' in kwargs.keys()):
        total_ticks = (x.max() + 1)/kwargs['xticks_dist']
        total_ticks = int(total_ticks)
        ax.set_xticks( np.linspace(0, kwargs['xticks_dist'] * total_ticks , total_ticks + 1, dtype = int ))
        ax.set_xticklabels(ax.get_xticks(), rotation = 90)
    
    
    max_y = max( simulated.max(), analytical.max() ) + max(simulated.std(), analytical.std() )
    ax.set_yticks(np.linspace(0, max_y , 11 ))
    ax.set_xlim(left = 0)
    plt.grid( linestyle = '--')
    plt.show()


def plot_pdf(distributions, lamda, mu, figsize = (10,5), **kwargs):
    pi = an.stationary_distribution(lamda, mu, capacity = distributions.index.max())
    pi = pi[:distributions.index.max() + 1]
    _kwargs = {
        'simulated_label' : 'simulated pdf',
        'analytical_label' : 'analytical pdf',
        'x_label' : 'queue_size',
         'y_label' : 'pdf'
    }
    kwargs.update(_kwargs)
    plot(distributions['pdf'], distributions['pdf_CI'], pi, figsize= figsize, **kwargs)

    
def plot_cdf(distributions, lamda, mu,figsize = (10,5), **kwargs):
    pi = an.stationary_distribution(lamda, mu, capacity = distributions.index.max())
    pi = pi.cumsum()
    pi = pi[:distributions.index.max() + 1]

    _kwargs = {
        'simulated_label' : 'simulated pdf',
        'analytical_label' : 'analytical pdf',
        'x_label' : 'queue_size',
         'y_label' : 'pdf'
    }
    kwargs.update(_kwargs)
    plot(distributions['cdf'], distributions['cdf_CI'], pi, figsize = figsize,**kwargs)
    
