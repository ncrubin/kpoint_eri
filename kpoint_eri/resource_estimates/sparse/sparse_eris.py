import numpy as np
from itertools import product

from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping

def build_eris_kpt(pyscf_mf,
                   mos_pqrs: np.ndarray,
                   kpt_pqrs: np.ndarray,
                   compact=False):
    """
    Compute (momentum conserving) kpoint-integrals (pkp qkq | rkr sks) block
    """
    nk = len(pyscf_mf.kpts)
    eri_pqrs = pyscf_mf.with_df.ao2mo(
            mos_pqrs,
            kpts=kpt_pqrs,
            compact=compact) / nk
    return eri_pqrs

def build_sparse_eris(
        pyscf_mf,
        localization='ibo',
        threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    # TODO: Do localization somewhere!
    momentum_map = build_momentum_transfer_mapping(pyscf_mf.cell,
                                                   pyscf_mf.kpts)
    num_kpoints = len(kpoints)
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    nmo_tot = sum(C.shape[1] for C in mo_coeffs)
    eris = np.zeros((nmo_tot,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk, dtype=np.int32) - nmo_pk[0]
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            eri_pqrs = build_eris_kpt(pyscf_mf, mos_pqrs, kpt_pqrs, compact=False)
            P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
            Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
            R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
            S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
            eris[P,Q,R,S] = eri_pqrs.reshape(mos_shape)

    eris[np.abs(eris) < threshold] = 0.0

    return eris

def count_number_of_non_zero_elements(pyscf_mf,
                                      localization='ibo',
                                      threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    momentum_map = build_momentum_transfer_mapping(pyscf_mf.cell,
                                                   pyscf_mf.kpts)
    num_kpoints = len(kpoints)
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    num_non_zero = 0
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
            eri_pqrs = build_eris_kpt(pyscf_mf, mos_pqrs, kpt_pqrs, compact=True)
            num_non_zero += sum(abs(eri_pqrs.ravel()) > threshold)

    return num_non_zero
