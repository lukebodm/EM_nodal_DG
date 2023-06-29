import numpy as np
from diff_matrices import Vandermonde1D

def Lift2D(N, Np, Nfaces, Nfp, r, s, Fmask, V):
    """function [LIFT] = Lift2D()
    Purpose  : Compute surface to volume lift term for DG formulation
    
    It allows the computation of the surface integral on the faces of the 
    element and the subsequent incorporation of that information into the 
    overall solution within the element."""

    Emat = np.zeros((int(Np), int(Nfaces*Nfp)))

    # Face 1
    faceR = r[Fmask[:, 0]]
    V1D = Vandermonde1D(N, faceR)
    massEdge1 = np.linalg.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:, 0], 0:Nfp] = massEdge1

    # Face 2
    faceR = r[Fmask[:, 1]]
    V1D = Vandermonde1D(N, faceR)
    massEdge2 = np.linalg.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:, 1], Nfp:2*Nfp] = massEdge2

    # Face 3
    faceS = s[Fmask[:, 2]]
    V1D = Vandermonde1D(N, faceS)
    massEdge3 = np.linalg.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:, 2], 2*Nfp:3*Nfp] = massEdge3

    # inv(mass matrix) * \I_n (L_i, L_j)_{edge_n}
    LIFT = np.dot(V, np.dot(V.T, Emat))

    return LIFT

