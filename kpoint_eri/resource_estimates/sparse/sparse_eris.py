import h5py
import numpy as np
import os
from itertools import product

from kpoint_eri.resource_estimates import utils

def build_eris_kpt(df,
                   mos_pqrs: np.ndarray,
                   kpt_pqrs: np.ndarray,
                   compact=False):
    """
    Compute (momentum conserving) kpoint-integrals (pkp qkq | rkr sks) block
    """
    nk = len(df.kpts)
    eri_pqrs = df.ao2mo(
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
    momentum_map = utils.build_momentum_transfer_mapping(pyscf_mf.cell,
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
            eri_pqrs = build_eris_kpt(pyscf_mf.with_df, mos_pqrs, kpt_pqrs, compact=False)
            P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
            Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
            R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
            S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
            eris[P,Q,R,S] = eri_pqrs.reshape(mos_shape)

    eris[np.abs(eris) < threshold] = 0.0

    return eris

def write_hamil_sparse(
        comm,
        pyscf_mf,
        filename='sparse.h5',
        localization='ibo',
        threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    # TODO: Do localization somewhere!
    momentum_map = utils.build_momentum_transfer_mapping(pyscf_mf.cell,
                                                   pyscf_mf.kpts)
    num_kpoints = len(kpoints)
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    nmo_tot = sum(C.shape[1] for C in mo_coeffs)
    eris = np.zeros((nmo_tot,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk, dtype=np.int32) - nmo_pk[0]
    base = filename.split('.')[0]
    rank = comm.rank
    tmp_file = f'tmp_{base}_{rank}.h5'
    fh5 = h5py.File(tmp_file, 'w')
    for iq, ikp, iks in product(range(num_kpoints), repeat=3):
        iproc_index = iq * num_kpoints**2 + ikp*num_kpoints + iks
        ikq = momentum_map[iq, ikp]
        ikr = momentum_map[iq, iks]
        kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
        mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
        mos_shape = [C.shape[1] for C in mos_pqrs]
        if iproc_index % comm.size == comm.rank:
            eri_pqrs = build_eris_kpt(pyscf_mf.with_df, mos_pqrs, kpt_pqrs, compact=False)
            eri_pqrs[abs(eri_pqrs) < threshold] = 0.0
            fh5[f'{iq}_{ikp}_{iks}'] = eri_pqrs

    fh5.close()
    comm.barrier()
    if comm.rank == 0:
        with h5py.File(filename, 'w') as fh5:
            for irank in range(comm.size):
                with h5py.File(f'tmp_{base}_{irank}.h5', 'r') as fh5_tmp:
                    for key in fh5_tmp.keys():
                        fh5[key] = fh5_tmp[key][:]
    comm.barrier()
    try:
        os.remove(tmp_file)
    except:
        print(f"Error removing {tmp_file} on {comm.rank}.")

def count_number_of_non_zero_elements(pyscf_mf,
                                      localization='ibo',
                                      threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    momentum_map = utils.build_momentum_transfer_mapping(pyscf_mf.cell,
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
            eri_pqrs = build_eris_kpt(pyscf_mf.with_df, mos_pqrs, kpt_pqrs, compact=True)
            num_non_zero += sum(abs(eri_pqrs.ravel()) > threshold)

    return num_non_zero
