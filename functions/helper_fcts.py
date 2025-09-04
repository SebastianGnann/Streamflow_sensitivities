import numpy as np
import matplotlib.pyplot as plt

# This script contains a collection of helper functions, e.g. to make plots.

def plot_origin_line(x, y, ax, **kwargs):

    lower_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='black', alpha=0.5, zorder=1)

    
def plot_Budyko_limits(x, y, ax, **kwargs):

    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', c='gray')
    ax.plot(np.linspace(1, lim, 100), np.linspace(1, 1, 100), '--', c='gray')
    ax.plot(np.linspace(0, lim, 100), np.linspace(0, 0, 100), '--', c='lightgray')
    
    #ax.fill_between(np.linspace(0,10,10), np.ones(10), 2*np.ones(10), color='lightgrey', alpha=0.25)
    ax.fill_between(np.linspace(0,10,10), -1*np.ones(10), 0*np.ones(10), color='lightgrey', alpha=0.25)
    ax.fill_between(np.concatenate((np.linspace(0,1,10), np.linspace(1,10,10))), 
                      2*np.ones(20), np.concatenate((np.linspace(0,1,10), np.ones(10))), color='lightgrey', alpha=0.25)
    
    
def plot_Budyko_curve(aridity, ax, **kwargs):
    
    evaporative_fraction = Budyko_curve(aridity)
    ax.plot(aridity,evaporative_fraction, 'k')
    
    
def Budyko_curve(aridity, **kwargs):
    
    # Budyko, M.I., Miller, D.H. and Miller, D.H., 1974. Climate and life (Vol. 508). New York: Academic press.
    return np.sqrt(aridity*np.tanh(1/aridity)*(1 - np.exp(-aridity)));