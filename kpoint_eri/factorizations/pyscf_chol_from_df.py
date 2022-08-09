from functools import reduce
import os
import numpy as np
from pyscf import gto as mol_gto
from pyscf.pbc import gto, scf, cc, df, mp
import h5py

from pyscf.gto.basis import parse_nwchem

from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf import lib

def cholesky_from_df_ints(mp):
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.
    Arguments:
        mp (KMP2) -- A KMP2 instance
    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts, nkpts, naux, nocc, nvir)
    """
    from pyscf.pbc.df import df
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(mp.stdout, mp.verbose)

    if mp._scf.with_df._cderi is None:
        mp._scf.with_df.build()

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    #nocc = mp.nocc
    nmo = mp.nmo
    # nvir = nmo - nocc
    nao = cell.nao_nr()

    mo_coeff = _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    kpts = mp.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Lchol = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())

    bra_start = 0
    bra_end = nmo
    ket_start = 0
    ket_end = nmo
    with h5py.File(mp._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'][:]
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
                Lchol[ki, kj] = out.reshape(-1, nmo, nmo)

    log.timer_debug1("transforming DF-AO integrals to MO", *cput0)

    return Lchol


def _format_jks(vj, dm, kpts_band):
    if kpts_band is None:
        vj = vj.reshape(dm.shape)
    elif kpts_band.ndim == 1:  # a single k-point on bands
        vj = vj.reshape(dm.shape)
    elif getattr(dm, "ndim", 0) == 2:
        vj = vj[0]
    return vj


if __name__ == '__main__':
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-cc-dzvp'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 0
    cell.build()

    name_prefix = ''
    basname = cell.basis
    pp_name = cell.pseudo

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)

    #FURTHER TESTING OF CHOLESKY INTEGRALS