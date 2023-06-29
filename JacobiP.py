import numpy as np
from scipy.special import gamma


def JacobiP(x, alpha, beta, N):
    xp = x if x.ndim == 1 else x.T
    PL = np.zeros((N + 1, len(xp)))

    gamma0 = (2 ** (alpha + beta + 1) / (alpha + beta + 1)) * \
             gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = (((alpha + beta + 2) * xp) / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :]

    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) /
                                      ((h1 + 1) * (h1 + 3)))
        bnew = -(alpha ** 2 - beta ** 2) / (h1 * (h1 + 2))
        PL[i + 1, :] = 1 / anew * (-aold * PL[i - 1, :] + (xp - bnew) * PL[i, :])
        aold = anew

    return PL[N, :]