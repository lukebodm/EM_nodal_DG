import numpy as np
from JacobiP import JacobiP
def GradJacobiP(r, alpha, beta, N):
    dP = np.zeros(len(r))
    if N == 0:
        dP[:] = 0.0
    else:
        #dP = np.sqrt(N*(N+alpha+beta+1)) * jacobi(N-1, alpha+1, beta+1)(r)
        dP = np.sqrt(N*(N+alpha+beta+1)) * JacobiP(r,alpha+1, beta+1, N-1)

    return dP

def GradSimplex2DP(a, b, id, jd):
    #fa = jacobi(id,0,0)(a)
    fa = JacobiP(a, 0, 0, id)
    dfa = GradJacobiP(a, 0, 0, id)
    #gb = jacobi(jd, 2*id+1,0)(b)
    gb = JacobiP(b, 2*id+1, 0, jd)
    dgb = GradJacobiP(b, 2*id+1, 0, jd)

    # r-derivative
    dmodedr = dfa * gb
    if id > 0:
        dmodedr = dmodedr * ((0.5*(1-b))**(id-1))

    # s-derivative
    dmodeds = dfa * (gb * (0.5*(1+a)))
    if id > 0:
        dmodeds = dmodeds * ((0.5*(1-b))**(id-1))

    tmp = dgb * ((0.5*(1-b))**id)
    if id > 0:
        tmp = tmp - 0.5*id*gb * ((0.5*(1-b))**(id-1))
    dmodeds = dmodeds + fa * tmp

    # Normalize
    dmodedr = 2**(id+0.5) * dmodedr
    dmodeds = 2**(id+0.5) * dmodeds

    return dmodedr, dmodeds


def GradVandermonde2D(N, r, s):
    V2Dr = np.zeros((len(r), int((N+1)*(N+2)/2)))
    V2Ds = np.zeros((len(r), int((N+1)*(N+2)/2)))

    # Find tensor-product coordinates
    a, b = rstoab(r, s)

    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk = sk + 1
    return V2Dr, V2Ds

def Dmatrices2D(N, r, s, V):
    Vr, Vs = GradVandermonde2D(N, r, s)
    Ds = np.matmul(Vs, np.linalg.inv(V))
    Dr = np.matmul(Vr, np.linalg.inv(V))
    return Dr, Ds

def rstoab(r, s):
    Np = len(r)
    a = np.zeros(Np)
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
        else:
            a[n] = -1
    b = s

    return a, b

def Simplex2DP(a, b, i, j):
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i+1,0, j)
    P = np.sqrt(2.0) * h1 * h2 * (1 - b)**i
    return P

def Vandermonde2D(N, r, s):
    V2D = np.zeros((len(r), int((N+1)*(N+2)/2)))

    # Transfer to (a,b) coordinates
    a, b = rstoab(r, s)

    # Build the Vandermonde matrix
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk = sk + 1

    return V2D


def Vandermonde1D(N, r):
    V1D = np.zeros((len(r), N+1))
    for j in range(1, N+2):
        V1D[:, j - 1] = JacobiP(r, 0, 0, j-1)
    return V1D



