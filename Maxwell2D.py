import matplotlib.pyplot as plt
import numpy as np
from MaxwellRHS2D import MaxwellRHS2D
from dtscale import dtscale2D
from JacobiGQ import JacobiGQ
import matplotlib
matplotlib.use('Agg')


def Maxwell2D(NODETOL, N, Np, Nfp, Nfaces, vmapM, vmapP, vmapB, mapB, K, x, y, r, s, nx, ny, Hx, Hy, Ez, FinalTime, Dr, Ds, rx, ry, sx, sy, LIFT, Fscale):
    """
    Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx,Hy,Ez
    """

    #Low storage Runge-Kutta coefficients
    rk4a = [0.0,
            -567301805773.0 / 1357537059087.0,
            -2404267990393.0 / 2016746695238.0,
            -3550918686646.0 / 2091501179385.0,
            -1275806237668.0 / 842570457699.0]

    rk4b = [1432997174477.0 / 9575080441755.0,
            5161836677717.0 / 13612068292357.0,
            1720146321549.0 / 2090206949498.0,
            3134564353537.0 / 4481467310338.0,
            2277821191437.0 / 14882151754819.0]

    rk4c = [0.0,
            1432997174477.0 / 9575080441755.0,
            2526269341429.0 / 6820363962896.0,
            2006345519317.0 / 3224310063776.0,
            2802321613138.0 / 2924317926251.0]

    time = 0
    # Runge-Kutta residual storage
    resHx = np.zeros((Np, K))
    resHy = np.zeros((Np, K))
    resEz = np.zeros((Np, K))

    # compute time step size
    rLGL, dummy = JacobiGQ(0, 0, N)
    rmin = abs(rLGL[0] - rLGL[1])
    dtscale = dtscale2D(NODETOL, x, y, r, s)
    dt = min(dtscale) * rmin * 2 / 3

    # outer time step loop
    i = 0
    while time < FinalTime:
        if time + dt > FinalTime:
            dt = FinalTime - time

        for INTRK in range(1, 6):
            # compute right hand side of TM-mode Maxwell's equations
            rhsHx, rhsHy, rhsEz = MaxwellRHS2D(Nfp, Nfaces, K, vmapM, vmapP, vmapB, mapB, Hx, Hy, Ez, nx, ny, Dr, Ds, rx, ry, sx, sy, LIFT, Fscale)

            # initiate and increment Runge-Kutta residuals
            resHx = rk4a[INTRK - 1] * resHx + dt * rhsHx
            resHy = rk4a[INTRK - 1] * resHy + dt * rhsHy
            resEz = rk4a[INTRK - 1] * resEz + dt * rhsEz

            # update fields
            Hx = Hx + rk4b[INTRK - 1] * resHx
            Hy = Hy + rk4b[INTRK - 1] * resHy
            Ez = Ez + rk4b[INTRK - 1] * resEz

        # Increment time
        time = time + dt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, Hx)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        file_name = 'images/' + str(i).zfill(4) + '.png'
        plt.savefig(file_name, dpi=1200)
        plt.close()

        i += 1
    return Hx, Hy, Ez, time
