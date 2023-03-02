import numpy as np
from pyscf.pbc import gto

def build_momentum_transfer_mapping(
        cell: gto.Cell,
        kpoints: np.ndarray
        ) -> np.ndarray:
    # Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    # k1 - k2 + G = Q.
    a = cell.lattice_vectors() / (2*np.pi)
    delta_k1_k2_Q = kpoints[:,None,None,:] - kpoints[None,:,None,:] - kpoints[None,None,:,:]
    delta_k1_k2_Q += kpoints[0][None, None, None, :]  # shift to center
    delta_dot_a = np.einsum('wx,kpQx->kpQw', a, delta_k1_k2_Q)
    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    mapping = np.where(np.sum(np.abs(delta_dot_a-int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,)*2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]

    return momentum_transfer_map