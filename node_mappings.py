import math
import numpy as np
from JacobiGQ import JacobiGQ
#from scipy.special import jacobi
from JacobiP import JacobiP
from diff_matrices import Vandermonde1D

def JacobiGL(alpha, beta, N):
    """
    function that computes the N'th order Gauss Lobatto quadrature points, x,
    associated with the Jacobi Polynomial of type (alpha, beta) > -1 ( <> -0.5)
    :param alpha:
    :param beta:
    :param N:
    :return:
    """
    x = np.zeros(N + 1)

    if N == 1:
        x[0] = -1.0
        x[1] = 1.0
        return x
    xint, w = JacobiGQ(alpha + 1, beta + 1, N - 2)
    x[0] = -1
    x[1:-1] = xint
    x[N] = 1

    return x

def Warpfactor(N, rout):
    """
    Compute scaled warp function at order N based on rout interpolation
    """

    # Compute LGL and equidistant node distribution (req)
    LGLr = JacobiGL(0, 0, N)
    req = np.linspace(-1, 1, N + 1)

    # Compute V based on req (to evaluate the polynomials at this set of points)
    Veq = Vandermonde1D(N, req)

    # Evaluate Lagrange polynomial at rout
    Nr = len(rout)
    Pmat = np.zeros((N + 1, Nr))
    for i in range(1, N + 1):
        Pmat[i - 1, :] = JacobiP(rout, 0, 0, i-1)
    Lmat = np.linalg.solve(Veq.T, Pmat)

    # Compute warp factor
    warp = np.dot(Lmat.T, (LGLr - req))

    # Scale factor
    zerof = np.abs(rout) < 1.0 - 1.0e-10
    sf = 1.0 - (zerof * rout) ** 2
    warp = warp / sf + warp * (zerof - 1)

    return warp


def GeometricFactors2D(x, y, Dr, Ds):
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    J = -xs * yr + xr * ys
    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J
    return rx, sx, ry, sy, J

def Nodes2D(N):
    """ function that creates a equidistant set of points on an equilateral triangle, 
    then uses a warp and blend function to make them into better interpolation points
    
    returns a set of points, (x,y) and the lambda values (L1, L2, L3)"""

    # optimized alpha values up to order 16
    alpopt = np.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                      1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258])

    # Set optimized parameter, alpha, depending on order N
    if N < 16:
        alpha = alpopt[N]
    else:
        alpha = 5/3

    # total number of nodes
    Np = int((N+1)*(N+2)/2)

    # Create equidistributed nodes on equilateral triangle
    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)

    sk = 0
    for n in range(1, N+2):
        for m in range(1, N+3-n):
            L1[sk] = (n-1)/N
            L3[sk] = (m-1)/N
            sk = sk + 1

    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2*L1) / np.sqrt(3.0)

    # Compute blending function at each node for each edge
    blend1 = 4 * L2 * L3
    blend2 = 4 * L1 * L3
    blend3 = 4 * L1 * L2

    # Amount of warp for each node, for each edge
    warpf1 = Warpfactor(N, L3 - L2)
    warpf2 = Warpfactor(N, L1 - L3)
    warpf3 = Warpfactor(N, L2 - L1)

    # Combine blend & warp
    warp1 = blend1 * warpf1 * (1 + (alpha * L1) ** 2)
    warp2 = blend2 * warpf2 * (1 + (alpha * L2) ** 2)
    warp3 = blend3 * warpf3 * (1 + (alpha * L3) ** 2)

    # Accumulate deformations associated with each edge
    x = x + 1 * warp1 + np.cos(2*np.pi/3) * warp2 + np.cos(4*np.pi/3) * warp3
    y = y + 0 * warp1 + np.sin(2*np.pi/3) * warp2 + np.sin(4*np.pi/3) * warp3

    return x, y

def xytors(x, y):
    L1 = (math.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - math.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - math.sqrt(3.0) * y + 2.0) / 6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s