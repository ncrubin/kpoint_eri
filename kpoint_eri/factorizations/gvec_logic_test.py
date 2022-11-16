import itertools
import numpy as np
from kpoint_eri.factorizations.gvec_logic import (get_miller_indices, 
get_delta_kp_kq_Q, build_transfer_map, build_G_vectors, build_gpq_mapping)

def test_get_miller_indices():
    kmesh = [3, 1, 1]
    int_scaled_kpts = get_miller_indices(kmesh)
    assert np.allclose(int_scaled_kpts[:, 0], np.arange(3))
    assert np.allclose(int_scaled_kpts[:, 1], 0)
    assert np.allclose(int_scaled_kpts[:, 2], 0)

    kmesh = [3, 2, 1]
    int_scaled_kpts = get_miller_indices(kmesh)
    assert np.allclose(int_scaled_kpts[:, 0], 
                       [0, 0, 1, 1, 2, 2])
    assert np.allclose(int_scaled_kpts[:, 1], 
                       [0, 1, 0, 1, 0, 1])

def test_get_delta_k1_k2_Q():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    for (kpidx, kqidx, qidx) in itertools.product(range(nkpts), repeat=3):
        assert np.allclose(scaled_kpts[kpidx] - scaled_kpts[kqidx] - scaled_kpts[qidx], 
                           delta_k1_k2_Q_int[kpidx, kqidx, qidx])

def test_transfer_map():
    kmesh = [3, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    true_transfer_map = np.array([[0, 1, 2],
                                  [2, 0 ,1],
                                  [1, 2, 0]
                                 ])
    assert np.allclose(transfer_map, true_transfer_map)

    kmesh = [4, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    true_transfer_map = np.array([[0, 1, 2, 3],
                                  [3, 0, 1, 2],
                                  [2, 3, 0, 1],
                                  [1, 2, 3, 0]
                                 ])
    assert np.allclose(transfer_map, true_transfer_map)

    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    true_transfer_map = np.array([[0, 1, 2, 3, 4, 5], 
                                  [1, 0, 3, 2, 5, 4],
                                  [4, 5, 0, 1, 2, 3],
                                  [5, 4, 1, 0, 3, 2],
                                  [2, 3, 4, 5, 0, 1],
                                  [3, 2, 5, 4, 1, 0]])
    assert np.allclose(transfer_map, true_transfer_map)

def test_build_Gvectors():
    g_dict = build_G_vectors()
    indx = 0
    for n1, n2, n3 in itertools.product(range(-1, 2), repeat=3):
        assert np.isclose(g_dict[(n1, n2, n3)], indx)
        indx += 1

def test_gpq_mapping():
    kmesh = [4, 3, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    gpq_map = build_gpq_mapping(kmesh, scaled_kpts)
    g_dict = build_G_vectors()
    g_dict_rev = dict(zip(g_dict.values(), g_dict.keys()))
    for iq in range(nkpts):
        for ikp in range(nkpts):
            ikq = transfer_map[iq, ikp]
            q = scaled_kpts[ikp] - scaled_kpts[ikq]
            # print(q / kmesh[0])
            # recall that gpq_map[iq, ikp] provides the G_{k,k-Q} index that we need to shift
            # Q back into the 1st Brillouin zone.
            miller_G_shift = g_dict_rev[gpq_map[iq, ikp]]
            # print(np.array(miller_G_shift))
            # print(scaled_kpts[iq] / kmesh[0] 
            #      +
            #     np.array(miller_G_shift)
            #     )
            # print()
            assert np.allclose(q, scaled_kpts[iq] + np.multiply(miller_G_shift, kmesh))

if __name__ == "__main__":
    test_get_miller_indices()
    test_get_delta_k1_k2_Q()
    test_transfer_map()
    test_build_Gvectors()
    test_gpq_mapping()