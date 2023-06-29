import numpy as np
from meshing_functions import MeshReaderGambit2D
from node_mappings import Nodes2D, xytors, GeometricFactors2D
from diff_matrices import Dmatrices2D, Vandermonde2D, GradVandermonde2D
from surface_ints import Lift2D
from normals import Normals2D
from connectivity_maps import tiConnect2D, BuildMaps2D
from Maxwell2D import Maxwell2D


def StartUp2D(NODETOL, N, Nv, VX, VY, K, EToV):

    # some important constants
    Nfp = N+1                 # number of nodes on a face
    Np = int((N+1)*(N+2)/2)   # number of nodes in each element
    Nfaces=3                  # number of faces in each element
    
    # compute nodal set
    x_tmp, y_tmp = Nodes2D(N)  # finds good interpolation points on eq tri. using warp and blend
    r, s = xytors(x_tmp,y_tmp) # maps x and y to r and s, points on the standard triangle |

    # create the computational components (Vandermond matrix, Mass Matrix, 
    # and differention matrices
    V = Vandermonde2D(N, r, s)
    invV = np.linalg.inv(V)
    MassMatrix = invV.T*invV
    Dr, Ds = Dmatrices2D(N,r,s,V)

    # Build coordinates of all the nodes
    va = EToV[:, 0].T - 1   # vector of first node in each element
    vb = EToV[:, 1].T - 1   # second node in each element
    vc = EToV[:, 2].T - 1   # third node in each element

    # Reshape the VX and VY arrays to match the required shape
    VX = VX.reshape(-1, 1)  # Shape: (146, 1)
    VY = VY.reshape(-1, 1)  # Shape: (146, 1)

    # Affine mapping from r,s set of well-behaved interpolation points,
    # to the computational points, x and y in the physical grid
    x = 0.5 * (-(r + s) * VX[va] + (1 + r) * VX[vb] + (1 + s) * VX[vc])  # Shape: (66, 1)
    y = 0.5 * (-(r + s) * VY[va] + (1 + r) * VY[vb] + (1 + s) * VY[vc])  # Shape: (66, 1)
    x = x.T
    y = y.T

    # Find all the nodes that lie on each edge
    # this is needed to calculate the surface integral
    fmask1 = np.where(np.abs(s+1) < NODETOL)[0]
    fmask2 = np.where(np.abs(r+s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r+1) < NODETOL)[0]
    Fmask = np.concatenate((fmask1, fmask2, fmask3)).T
    Fx = x[Fmask, :]
    Fy = y[Fmask, :]

    # change the form of Fmask to work with later arrays
    Fmask = np.column_stack((fmask1, fmask2, fmask3))

    # Create surface integral terms
    # allows the computation of the surface integral on the faces of the 
    # element and the subsequent incorporation of that information into the 
    # overall solution within the element.
    LIFT = Lift2D(N, Np, Nfaces, Nfp, r, s, Fmask, V)

    # calculate geometric factors
    # nx(i,k) is the n^{hat} component of normal at face noide i on element k
    # J = Volume Jacobian
    # sJ = Jacobian at surface nodes
    # Fscale(i,k) = ratio of surface to volume Jacobian of face i on element k
    rx, sx, ry, sy, J = GeometricFactors2D(x, y, Dr, Ds) # metric constants
    nx, ny, sJ = Normals2D(K, Nfp, x, y, Dr, Ds, Fmask)
    Fscale = sJ / J[Fmask.flatten()]

    # build connectivity matrix
    EToE, EToF = tiConnect2D(EToV)

    # build connectivity maps
    mapM, mapP, vmapM, vmapP, vmapB, mapB = BuildMaps2D(NODETOL, K, Np, Nfp, Nfaces, Fmask, EToV, EToE, EToF, VX, VY, x, y)
    Vr, Vs = GradVandermonde2D(N, r, s)
    Drw = np.dot(np.dot(V, Vr.T), np.linalg.inv(np.dot(V, V.T)))
    Dsw = np.dot(np.dot(V, Vs.T), np.linalg.inv(np.dot(V, V.T)))

    return Np, Nfp, Nfaces, K, vmapM, vmapP, vmapB, mapB, x, y, r, s, nx, ny, rx, sx, ry, sy, Dr, Ds, LIFT, Fscale

if __name__ == "__main__":
    N = 10          # polynomial order used for approximation
    NODETOL = 1e-12 # used to find nodes on the edge
    
    # read in mesh from file
    Nv, VX, VY, K, EToV = MeshReaderGambit2D('Maxwell025.neu')

    # run startup script to find all computational elements needed for the time loop
    Np, Nfp, Nfaces, K, vmapM, vmapP, vmapB, mapB, x, y, r, s, nx, ny, rx, sx, ry, sy, Dr, Ds, LIFT, Fscale = StartUp2D(NODETOL, N, Nv, VX, VY, K, EToV)

    # set initial conditions
    mmode = 1
    nmode = 1
    Ez = np.sin(mmode * np.pi * x) * np.sin(nmode * np.pi * y)
    Hx = np.zeros((Np, K))
    Hy = np.zeros((Np, K))

    # solve problem
    FinalTime = 1
    Hx,Hy,Ez,time = Maxwell2D(NODETOL, N, Np, Nfp, Nfaces, vmapM, vmapP, vmapB, mapB, K, x, y, r, s, nx, ny, Hx, Hy, Ez, FinalTime, Dr, Ds, rx, ry, sx, sy, LIFT, Fscale)
