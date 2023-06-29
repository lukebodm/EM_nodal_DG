import numpy as np

def Normals2D(K, Nfp, x, y, Dr, Ds, Fmask):
    """function that computes outward pointing normals at element faces
    and surface Jacobians"""
    xr = Dr @ x
    yr = Dr @ y
    xs = Ds @ x
    ys = Ds @ y
    J = xr * ys - xs * yr

    fxr = xr[Fmask.flatten()]
    fxs = xs[Fmask.flatten()]
    fyr = yr[Fmask.flatten()]
    fys = ys[Fmask.flatten()]

    nx = np.zeros((3 * Nfp, K))
    ny = np.zeros((3 * Nfp, K))
    sJ = np.zeros((Nfp, K))

    fid1 = np.arange(0, Nfp)
    fid2 = np.arange(Nfp, 2 * Nfp)
    fid3 = np.arange(2 * Nfp, 3 * Nfp)

    #fid1 = list(fid1)
    #fid2 = list(fid2)
    #fid3 = list(fid3)

    # face 1
    #breakpoint()
    nx[fid1] = fyr[fid1]
    ny[fid1] = -fxr[fid1]

    # face 2
    nx[fid2] = fys[fid2] - fyr[fid2]
    ny[fid2] = -fxs[fid2] + fxr[fid2]

    # face 3
    nx[fid3] = -fys[fid3]
    ny[fid3] = fxs[fid3]

    # normalize
    sJ = np.sqrt(nx * nx + ny * ny)
    nx = nx / sJ
    ny = ny / sJ

    return nx, ny, sJ

