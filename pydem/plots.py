from numpy import *
from pylab import *
import h5py
import base64
from matplotlib import animation
from IPython.display import display, HTML
from ipywidgets import FloatProgress

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

def make_video(fname, show=True, figsize=(12,8), vmax=None, cmap='inferno'):
    with h5py.File('{0}.h5'.format(fname), 'r') as f:
        time = f['time'][:]
        n = time.shape[0]

        if vmax is None:
            vmax = norm(f['step-0/vel'][:],axis=1).max()

        bar = FloatProgress(min=0, max=n-1, description='frames: {0}'.format(n))
        display(bar)

        def make_frame(step):
            coo = f['step-{0}/coo'.format(step)][:]
            vel = f['step-{0}/vel'.format(step)][:]
            L,H = amax(coo, axis=0)

            gcf().clear()
            dots = scatter(coo[:,0], coo[:,1], s=9, c=norm(vel,axis=1),
                    vmax=vmax, linewidth=0, cmap=cmap)

            m = L * 0.01
            xlim([-m, L+m])
            ylim([-m, H+m])
            colorbar()
            gca().set_aspect('equal')

            bar.value = step
            bar.description = '{0}/{1}'.format(step, n)
            return dots

        fig = figure(figsize=figsize);
        anim = animation.FuncAnimation(fig, make_frame, frames=n, interval=30)
        anim.save('{0}.mp4'.format(fname), bitrate=3200, extra_args=['-vcodec', 'libx264'])
        close()

        if show:
            return HTML("""
            <video width=800 controls>
            <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
            </video>
            """.format(base64.encodebytes(open('{0}.mp4'.format(fname), 'rb').read()).decode()))
