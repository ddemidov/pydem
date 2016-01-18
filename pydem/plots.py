from numpy import *
from pylab import *

def plot_potentials(*potentials, fs=None):
    figure(figsize=fs)

    x = linspace(0.5, 4, 100)

    for p in potentials:
        plot(x, p(x))

    plot(x, zeros_like(x), ':k')
    ylim([-5, 5])

def plot_points(coo, color, fs=None):
    figure(figsize=fs)
    L,H = amax(coo, axis=0)

    scatter(coo[:,0], coo[:,1], s=9, c=color, linewidth=0, cmap='inferno');

    m = L * 0.01
    xlim([-m, L+m])
    ylim([-m, H+m])
    colorbar()

    gca().set_aspect('equal')
