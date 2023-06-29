import numpy as np

def BuildMaps2D(NODETOL, K, Np, Nfp, Nfaces, Fmask, EToV, EToE, EToF, VX, VY, x, y):

    # number volume nodes consecutively
    nodeids = np.reshape(np.arange(1, int(K*Np+1)), (int(Np), K), order='F')
    vmapM = np.zeros((Nfp, Nfaces, K))
    vmapP = np.zeros((Nfp, Nfaces, K))
    mapM = np.arange(1, K*Nfp*Nfaces+1)
    mapP = np.reshape(mapM, (Nfp, Nfaces, K), order='F')

    # find index of face nodes with respect to volume node ordering
    for k1 in range(K):
        for f1 in range(Nfaces):
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]

    one = np.ones(Nfp)
    for k1 in range(K):
        for f1 in range(Nfaces):
            # find neighbor
            k2 = EToE[k1, f1] - 1
            f2 = EToF[k1, f1] - 1

            # reference length of edge
            v1 = EToV[k1, f1]
            v2 = EToV[k1, (f1+1) % Nfaces]
            refd = np.sqrt((VX[v1-1] - VX[v2-1])**2 + (VY[v1-1] - VY[v2-1])**2)

            # find volume node numbers of left and right nodes
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            x1 = np.ravel(x, order='F')[vidM.astype(int) - 1]
            y1 = np.ravel(y, order='F')[vidM.astype(int) - 1]
            x2 = np.ravel(x, order='F')[vidP.astype(int) - 1]
            y2 = np.ravel(y, order='F')[vidP.astype(int) - 1]
            x1 = np.outer(x1, one)
            y1 = np.outer(y1, one)
            x2 = np.outer(x2, one)
            y2 = np.outer(y2, one)

            # Compute distance matrix
            D = (x1 - x2.T) ** 2 + (y1 - y2.T) ** 2
            idP, idM = np.where(np.sqrt(np.abs(D)) < NODETOL * refd)

            vmapP[idM, f1, k1] = vidP[idP]
            mapP[idM, f1, k1] = (idP + f2 * Nfp + k2 * Nfaces * Nfp) + 1

    # reshape vmapM and vmapP to be vectors and create boundary node list
    vmapP = np.ravel(vmapP, order='F')
    vmapM = np.ravel(vmapM, order='F')
    mapP = np.ravel(mapP, order='F')
    mapB = np.where(vmapP == vmapM)[0] + 1
    vmapB = vmapM[mapB-1]

    return mapM, mapP, vmapM, vmapP, vmapB, mapB


def tiConnect2D(EToV):
    """
    triangle face connect algorithm

    :param EToV:
    :return: EToE(k,i) global element number to which face i of element k connects
             EToF(k,i) - the local face number to which face i ok element k connects
    """
    Nfaces = 3
    K = EToV.shape[0]
    Nnodes = np.max(EToV)

    # create list of all faces 1, then 2, & 3
    fnodes = np.vstack((EToV[:, [0, 1]], EToV[:, [1, 2]], EToV[:, [2, 0]]))
    fnodes = np.sort(fnodes, axis=1) - 1

    # set up default element to element and Element to faces connectivity
    EToE = np.tile(np.arange(1,K+1).reshape(-1, 1), (1, Nfaces))
    EToF = np.tile(np.arange(1,Nfaces+1), (K, 1))

    # uniquely number each set of three faces by their node numbers
    id = fnodes[:, 0] * Nnodes + fnodes[:, 1] + 1
    spNodeToNode = np.column_stack((id, np.arange(1, Nfaces * K + 1), np.ravel(EToE, order='F'), np.ravel(EToF, order='F')))

    # Now we sort by global face number.
    sorted_spNode = np.array(sorted(spNodeToNode, key=lambda x: (x[0], x[1])))

    # find matches in the sorted face list
    matches = np.where(sorted_spNode[:-1, 0] == sorted_spNode[1:, 0])[0]

    # make links reflexive
    matchL = np.vstack((sorted_spNode[matches], sorted_spNode[matches + 1]))
    matchR = np.vstack((sorted_spNode[matches + 1], sorted_spNode[matches]))

    # insert matches
    EToE_tmp = np.ravel(EToE, order='F') - 1
    EToF_tmp = np.ravel(EToF, order='F') - 1

    EToE_tmp[matchL[:, 1] - 1] = (matchR[:, 2] - 1)
    EToF_tmp[matchL[:, 1] - 1] = (matchR[:, 3] - 1)

    EToE = EToE_tmp.reshape(EToE.shape, order='F') + 1
    EToF = EToF_tmp.reshape(EToF.shape, order='F') + 1

    return EToE, EToF

