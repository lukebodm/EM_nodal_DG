import numpy as np

def MeshReaderGambit2D(FileName):
    """
    Read in basic grid information to build grid
    NOTE: gambit(Fluent, Inc) *.neu format is assumed
    """

    # Open the file
    with open(FileName, 'r') as Fid:
        # Read intro
        for i in range(6):
            line = Fid.readline()

        # Find number of nodes and number of elements
        dims = list(map(int, Fid.readline().split()))
        Nv = dims[0]
        K = dims[1]

        # move down to lines in file
        for i in range(2):
            line = Fid.readline()

        # Read node coordinates
        VX = [0.0] * Nv
        VY = [0.0] * Nv
        for i in range(Nv):
            line = Fid.readline()
            tmpx = list(map(float, line.split()))
            VX[i] = tmpx[1]
            VY[i] = tmpx[2]

        # move down two lines in file
        for i in range(2):
            line = Fid.readline()

        # Read element to node connectivity
        EToV = [[0] * 3 for _ in range(K)]
        for k in range(K):
            line = Fid.readline()
            tmpcon = list(map(float, line.split()))
            EToV[k][0] = int(tmpcon[3])
            EToV[k][1] = int(tmpcon[4])
            EToV[k][2] = int(tmpcon[5])

        # Return the results
        EToV = np.array(EToV)
        VX = np.array(VX)
        VY = np.array(VY)


    return Nv, VX, VY, K, EToV


