import itertools
import numpy as np
def cartesian_prod(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian_prod(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    '''
    arrays = [np.asarray(x) for x in arrays]
    dtype = np.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])

    return out.reshape(nd,-1).T

def get_miller_indices(kmesh):
    """
    Calculate the miller indices on a gamma centered non-1stBZ Monkorhst-Pack mesh

    :param kmesh: 1-D iteratble with the number of k-points in the x,y,z direction
                  [Nk_x, NK_y, NK_z] where NK_x/y/z are positive integers
    :returns: np.array 2D that is prod([Nk_x, NK_y, NK_z])
    """
    if kmesh[0] < 1:
        raise TypeError("Bad miller index dimension in x")
    if kmesh[1] < 1:
        raise TypeError("Bad miller index dimension in y")
    if kmesh[2] < 1:
        raise TypeError("Bad miller index dimension in z")

    ks_int_each_axis = []
    for n in kmesh:
        ks = np.arange(n, dtype=float) / n
        ks_int_each_axis.append(np.arange(n, dtype=float))
    int_scaled_kpts = cartesian_prod(ks_int_each_axis)
    return int_scaled_kpts


def get_delta_kp_kq_Q(int_scaled_kpts):
    """
    Generate kp - kq + G = Q as kp - kq - Q = G


    :param int_scaled_kpts: array of kpts represented as miller indices [[nkx, nky, nkz], ...]
    :returns: np.array nkpts x nkpts that corresponds to 
    """
    delta_k1_k2_Q_int = int_scaled_kpts[:, None, None, :] - int_scaled_kpts[None, :, None, :] - int_scaled_kpts[None, None, :, :]
    return delta_k1_k2_Q_int

def build_transfer_map(kmesh, scaled_kpts):
    """
    Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    k1 - k2 + G = Q.

    :param kmesh: kmesh [nkx, nky, nkz] the number of kpoints in each direction
    :param scaled_kpts: miller index representation [[0, nkx-1], [0, nky-1], [0, nkz-1]]
                        of all the kpoints
    :returns: transfer map satisfying k1 - k2 + G = Q in matrix form map[Q, k1] = k2
    """
    nkpts = len(scaled_kpts)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = np.zeros((nkpts, nkpts), dtype=np.int32)
    for (kpidx, kqidx, qidx) in itertools.product(range(nkpts), repeat=3):
        # explicitly build my transfer matrix
        if np.allclose([np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][0]) % kmesh[0],
                        np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][1]) % kmesh[1],
                        np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][2]) % kmesh[2],
                       ], 0
                      ):
            transfer_map[kqidx, kpidx] = qidx
    return transfer_map

def build_G_vectors():
    """Build all 27 Gvectors

    :param kmesh:     
    :returns tuple: G_dict a dictionary mapping miller index to appropriate
        G_vector index.  The actual cell Gvector can be recovered with 
        np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors()
    """
    G_dict = {}
    G_vectors = np.zeros((27, 3), dtype=np.float64)
    indx = 0
    for n1, n2, n3 in itertools.product(range(-1, 2), repeat=3):
        G_dict[(n1, n2, n3)] = indx
        # G_vectors[indx] = np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors())
        # miller_indx = np.rint(
        #     np.einsum("wx,x->w", lattice_vectors, G_vectors[indx]) / (2 * np.pi)
        # )
        # assert (miller_indx == (n1, n2, n3)).all()
        indx += 1
    return G_dict# , G_vectors

def build_gpq_mapping(kmesh, int_scaled_kpts):
    momentum_map = build_transfer_map(kmesh, int_scaled_kpts)
    nkpts = len(int_scaled_kpts)
    g_dict = build_G_vectors()
    Gpq_mapping = np.zeros((nkpts, nkpts), dtype=np.int32)
    for iq in range(nkpts):
        for ikp in range(nkpts):
            ikq = momentum_map[iq, ikp]
            delta_Gpq = (int_scaled_kpts[ikp] - int_scaled_kpts[ikq]) - int_scaled_kpts[iq]
            delta_Gpq[0] /= kmesh[0]
            delta_Gpq[1] /= kmesh[1]
            delta_Gpq[1] /= kmesh[2]
            miller_indx = np.rint(delta_Gpq)
            Gpq_mapping[iq, ikp] = g_dict[tuple(miller_indx)]
            
    return Gpq_mapping



