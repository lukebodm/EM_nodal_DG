import numpy as np
from scipy.special import gamma
from scipy.linalg import eig

def JacobiGQ(alpha, beta, N):
    """
    Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    """

    if N == 0:
        x = np.array([-(alpha-beta)/(alpha+beta+2)])
        w = np.array([2.0])
        return x, w

    # Form symmetric matrix from recurrence.
    h1 = 2*np.arange(N+1) + alpha + beta
    J = np.diag(-1/2*(alpha**2-beta**2)/(h1+2)/h1) + \
        np.diag(2/(h1[0:N]+2)*np.sqrt(np.arange(1, N+1)*((np.arange(1, N+1)+alpha+beta)*((np.arange(1, N+1)+alpha)*((np.arange(1, N+1)+beta)/(h1[0:N]+1))/(h1[0:N]+3)))), 1)
    if alpha + beta < 10*np.finfo(float).eps:
        J[0, 0] = 0.0
    J = J + J.T

    # Compute quadrature by eigenvalue solve
    D, V = eig(J)
    sorted_indices = np.argsort(D)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    w = (V[0, :]**2)*2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    return D, w

