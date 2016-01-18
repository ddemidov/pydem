import h5py
from numpy import *

def make_box(L, H, h):
    return vstack((
        mgrid[0:L+h/2:h, 0:h/2:h  ].transpose((2,1,0)).reshape((-1,2)),
        mgrid[0:L+h/2:h, H:H+h/2:h].transpose((2,1,0)).reshape((-1,2)),
        mgrid[0:h/2:h,   h:H-h/2:h].transpose((2,1,0)).reshape((-1,2)),
        mgrid[L:L+h/2:h, h:H-h/2:h].transpose((2,1,0)).reshape((-1,2)),
        ))

def pull_down(v):
    return '(double2)(0.0, -{0:e})'.format(v)

def checkpoint(fname, time, coo, vel, acc, typ, mass, radius, scale):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('time',   data=array([time]))
        f.create_dataset('coo',    data=coo)
        f.create_dataset('vel',    data=vel)
        f.create_dataset('acc',    data=acc)
        f.create_dataset('typ',    data=typ)
        f.create_dataset('mass',   data=mass)
        f.create_dataset('radius', data=radius)
        f.create_dataset('scale',  data=scale)

def restore(fname):
    with h5py.File(fname, 'r') as f:
        return (
            f['time'][0],
            f['coo'][:],
            f['vel'][:],
            f['acc'][:],
            f['typ'][:],
            f['mass'][:],
            f['radius'][:],
            f['scale'][:]
            )

def pickup(fname):
    with h5py.File(fname, 'r') as f:
        time = f['time'][:]
        nsteps = time.shape[0]

        coo = f['step-{0}/coo'.format(nsteps-1)][:]
        vel = f['step-{0}/vel'.format(nsteps-1)][:]
        acc = f['step-{0}/acc'.format(nsteps-1)][:]

        typ    = f['type'][:]
        mass   = f['mass'][:]
        radius = f['radius'][:]
        scale  = f['scale'][:]

    return (time[-1], coo, vel, acc, typ, mass, radius, scale)
