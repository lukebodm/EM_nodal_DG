import numpy as np
from vector_operators import Grad2D, Curl2D
def MaxwellRHS2D(Nfp, Nfaces, K, vmapM, vmapP, vmapB, mapB, Hx, Hy, Ez, nx, ny, Dr, Ds, rx, ry, sx, sy, LIFT, Fscale):
    """
    Evaluate RHS flux in 2D Maxwell TM form
    """

    Ez_shape = np.shape(Ez)

    # Define field differences at faces
    dHx = np.zeros((Nfp*Nfaces*K))
    #dHx = np.ravel(dHx, order='F')
    dHx = (np.ravel(Hx, order='F')[vmapM.astype(int)-1]-np.ravel(Hx, order='F')[vmapP.astype(int)-1])#.reshape(dHx.shape, order='F')

    dHy = np.zeros(Nfp*Nfaces*K)
    #dHy = np.ravel(dHy, order='F')
    dHy = (np.ravel(Hy, order='F')[vmapM.astype(int)-1]-np.ravel(Hy, order='F')[vmapP.astype(int)-1])#.reshape(dHy.shape, order='F')

    dEz = np.zeros(Nfp*Nfaces*K)
    #dEz = np.ravel(dEz, order='F')
    dEz = np.ravel(Ez, order='F')[vmapM.astype(int)-1]-np.ravel(Ez, order='F')[vmapP.astype(int)-1]#.reshape(dEz.shape, order='F')


    #dEz = dEz.reshape((Nfp * Nfaces, K), order='F')


    # Impose reflective boundary conditions (Ez+ = -Ez-)
    #dHx = np.ravel(dHx, order='F')
    #dHy = np.ravel(dHy, order='F')
    #dEz = np.ravel(dEz, order='F')
    Ez = np.ravel(Ez, order='F')

    dHx[mapB-1] = 0
    dHy[mapB-1] = 0
    dEz[mapB-1] = 2*Ez[vmapB.astype(int)-1]

    dHx = dHx.reshape((Nfp*Nfaces, K), order='F')
    dHy = dHy.reshape((Nfp*Nfaces, K), order='F')
    dEz = dEz.reshape((Nfp*Nfaces, K), order='F')
    Ez = Ez.reshape(Ez_shape, order='F')

    # Evaluate upwind fluxes
    alpha = 1.0
    ndotdH = nx*dHx + ny*dHy
    fluxHx = ny*dEz + alpha*(ndotdH*nx - dHx)
    fluxHy = -nx*dEz + alpha*(ndotdH*ny - dHy)
    fluxEz = -nx*dHy + ny*dHx - alpha*dEz

    # Local derivatives of fields
    Ezx, Ezy = Grad2D(Dr, Ds, rx, ry, sx, sy, Ez)
    CuHx, CuHy, CuHz = Curl2D(Dr, Ds, rx, ry, sx, sy, Hx, Hy, uz=None)

    # Compute right hand sides of the PDE's
    rhsHx = -Ezy + LIFT.dot((Fscale*fluxHx)/2.0)
    rhsHy = Ezx + LIFT.dot((Fscale*fluxHy)/2.0)
    rhsEz = CuHz + LIFT.dot((Fscale*fluxEz)/2.0)
    return rhsHx, rhsHy, rhsEz

