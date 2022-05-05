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
    mos_shape = [C.shape[1] for C in mos_pqrs]
    eri_pqrs = pyscf_mf.with_df.ao2mo(
            mos_pqrs,
            kpts=kpt_pqrs,
            compact=compact)
    return eri_pqrs

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
