import numpy as np

def dtscale2D(NODETOL, x, y, r, s):
    # Compute inscribed circle diameter as characteristic
    # for grid to choose timestep

    # Find vertex nodes
    vmask1 = np.where(np.abs(s+r+2) < NODETOL)[0]
    vmask2 = np.where(np.abs(r-1) < NODETOL)[0]
    vmask3 = np.where(np.abs(s-1) < NODETOL)[0]
    vmask = np.concatenate((vmask1, vmask2, vmask3))

    vx = x[vmask, :]
    vy = y[vmask, :]

    # Compute semi-perimeter and area
    len1 = np.sqrt((vx[0, :]-vx[1, :])**2 + (vy[0, :]-vy[1, :])**2)
    len2 = np.sqrt((vx[1, :]-vx[2, :])**2 + (vy[1, :]-vy[2, :])**2)
    len3 = np.sqrt((vx[2, :]-vx[0, :])**2 + (vy[2, :]-vy[0, :])**2)
    sper = (len1 + len2 + len3)/2.0
    Area = np.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3))

    # Compute scale using radius of inscribed circle
    dtscale = Area/sper
    return dtscale

