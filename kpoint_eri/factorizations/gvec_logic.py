import itertools
import numpy as np


def cartesian_prod(arrays, out=None):
    """
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

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = np.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[: nd - i])

    return out.reshape(nd, -1).T


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
    Generate kp - kq - Q = S for kp, kq, and Q.  The difference of the three integers
    is stored as a four tensor D_{kp, kq, Q} = S. where the dimension of D is
    (nkpts, nkpts, nkpts, 3).  The last dimension stores the x,y,z components of S.

    :param int_scaled_kpts: array of kpts represented as miller indices [[nkx, nky, nkz], ...]
    :returns: np.array nkpts x nkpts that corresponds to
    """
    delta_k1_k2_Q_int = (
        int_scaled_kpts[:, None, None, :]
        - int_scaled_kpts[None, :, None, :]
        - int_scaled_kpts[None, None, :, :]
    )
    return delta_k1_k2_Q_int


def build_transfer_map(kmesh, scaled_kpts):
    """
    Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    k1 - k2 + G = Q.
    where k1, k2, Q are all tuples of integers [[0, Nkx-1], [0, Nky-1], [0, Nkz-1]]
    and G is [{0, Nkx}, {0, Nky}, {0, Nkz}].

    This is computed from `get_delta_kp_kq_Q` which computes k1 - k2 -Q = S.
    Thus k1 - k2 = Q + S which shows that S is [{0, Nkx}, {0, Nky}, {0, Nkz}].

    Thus to compute map[Q, k1] = k2

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
        if np.allclose(
            [
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][0]) % kmesh[0],
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][1]) % kmesh[1],
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][2]) % kmesh[2],
            ],
            0,
        ):
            transfer_map[qidx, kpidx] = kqidx
    return transfer_map


def build_conjugate_map(kmesh, scaled_kpts):
    """
    build mapping that map[k1] = -k1
    """
    nkpts = len(scaled_kpts)
    kpoint_dict = dict(
        zip([tuple(map(int, scaled_kpts[x])) for x in range(nkpts)], range(nkpts))
    )
    kconj_map = np.zeros((nkpts), dtype=int)
    for kidx in range(nkpts):
        negative_k_scaled = -scaled_kpts[kidx]
        fb_negative_k_scaled = tuple(
            (
                int(negative_k_scaled[0]) % kmesh[0],
                int(negative_k_scaled[1]) % kmesh[1],
                int(negative_k_scaled[2]) % kmesh[2],
            )
        )
        kconj_map[kidx] = kpoint_dict[fb_negative_k_scaled]
    return kconj_map


def build_G_vectors(kmesh):
    """Build all 8 Gvectors

    :param kmesh:
    :returns tuple: G_dict a dictionary mapping miller index to appropriate
        G_vector index.  The actual cell Gvector can be recovered with
        np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors()
    """
    G_dict = {}
    G_vectors = np.zeros((8, 3), dtype=np.float64)
    indx = 0
    for n1, n2, n3 in itertools.product([0, -1], repeat=3):
        G_dict[(n1 * kmesh[0], n2 * kmesh[1], n3 * kmesh[2])] = indx
        indx += 1
    return G_dict


def build_gpq_mapping(kmesh, int_scaled_kpts):
    """
    build map for kp - kq = Q + G where G is [{0, -Nkx}, {0, -Nky}, {0, -Nkz}] .
    G will be 0 or Nkz because kp - kq takes on values between [-Nka + 1, Nka - 1]
    in each component.

    :param kmesh: number of k-points along each direction [Nkx, Nky, Nkz].
    :param int_scaled_kpts: scaled_kpts. Each kpoint is a tuple of 3 integers
                            where each integer is between [0, Nka-1].
    :returns: array mapping where first two indices are the index of kp and kq
              and the last dimension holds the gval that is [{0, Nkx}, {0, Nky}, {0, Nkz}].
    """
    momentum_map = build_transfer_map(kmesh, int_scaled_kpts)
    nkpts = len(int_scaled_kpts)
    g_dict = build_G_vectors(kmesh)
    Gpq_mapping = np.zeros((nkpts, nkpts, 3), dtype=np.int32)
    for iq in range(nkpts):
        for ikp in range(nkpts):
            ikq = momentum_map[iq, ikp]
            q_minus_g = int_scaled_kpts[ikp] - int_scaled_kpts[ikq]
            g_val = (
                0 if q_minus_g[0] >= 0 else -kmesh[0],
                0 if q_minus_g[1] >= 0 else -kmesh[1],
                0 if q_minus_g[2] >= 0 else -kmesh[2],
            )
            Gpq_mapping[ikp, ikq, :] = np.array(g_val)

    return Gpq_mapping


def compliment_g(g_val, q_val, kmesh, scaled_kpts):
    """
    Computes the compliment of g_val given q_val as

    !G1 = -(Q + G1 + (-Q)).
    """
    nkpts = len(scaled_kpts)
    kpoint_dict = dict(
        zip([tuple(map(int, scaled_kpts[x])) for x in range(nkpts)], range(nkpts))
    )
    conj_map = build_conjugate_map(kmesh, scaled_kpts)
    # compliment = -(q_val + g_val + -q_val)
    qidx = kpoint_dict[tuple(map(int, q_val))]
    complment_g_val = -(q_val + g_val + scaled_kpts[conj_map[qidx]])
