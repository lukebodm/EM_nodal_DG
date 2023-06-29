import numpy as np

def Grad2D(Dr, Ds, rx, ry, sx, sy, u):
    ur = np.matmul(Dr, u)
    us = np.matmul(Ds, u)
    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    return ux, uy

import numpy as np

def Curl2D(Dr, Ds, rx, ry, sx, sy, ux, uy, uz=None):
    uxr = np.matmul(Dr, ux)
    uxs = np.matmul(Ds, ux)
    uyr = np.matmul(Dr, uy)
    uys = np.matmul(Ds, uy)
    vz = rx * uyr + sx * uys - ry * uxr - sy * uxs
    vx = None
    vy = None

    if uz is not None:
        uzr = np.matmul(Dr, uz)
        uzs = np.matmul(Ds, uz)
        vx = ry * uzr + sy * uzs
        vy = -rx * uzr - sx * uzs

    return vx, vy, vz
