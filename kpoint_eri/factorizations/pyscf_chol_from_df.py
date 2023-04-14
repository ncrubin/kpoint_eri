import h5py
import numpy as np
import numpy.typing as npt

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc import mp


def cholesky_from_df_ints(mf, pad_mos_with_zeros: bool = True) -> npt.NDArray:
    """Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals. Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Args:
        mf: pyscf K-RMHF object
        pad_mos_with_zeros: Whether to follow KMP2 class and pad mo coefficients
            with (Default value = True) if system has varying number of
            occupied orbitals.

    Returns:
        Lchol: 3-center DF ints, with shape (nkpts, nkpts, naux, nmo, nmo)
    """

    log = logger.Logger(mf.stdout, mf.verbose)

    if mf.with_df._cderi is None:
        mf.with_df.build()

    cell = mf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    # nvir = nmo - nocc
    nao = cell.nao_nr()

    if pad_mos_with_zeros:
        mp_inst = mp.KMP2(mf)
        mo_coeff = _add_padding(mp_inst, mp_inst.mo_coeff, mp_inst.mo_energy)[0]
        nmo = mp_inst.nmo
    else:
        mo_coeff = mf.mo_coeff
        nmo = nao
        num_mo_per_kpt = np.array([C.shape[-1] for C in mo_coeff])
        err_msg = "Number of MOs differs at each k-point or is not the same as the number of AOs."
        assert (num_mo_per_kpt == nmo).all(), err_msg
    kpts = mf.kpts
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
    ket_start = nmo
    ket_end = 2 * nmo
    with h5py.File(mf.with_df._cderi, "r") as f:
        kptij_lst = f["j3c-kptij"][:]
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, "j3c", kpti_kptj, kptij_lst))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order="F")
                if dtype == np.double:
                    out = _ao2mo.nr_e2(
                        Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym="s2"
                    )
                else:
                    # Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(
                        Lpq_ao,
                        mo,
                        (bra_start, bra_end, ket_start, ket_end),
                        tao,
                        ao_loc,
                    )
                Lchol[ki, kj] = out.reshape(-1, nmo, nmo)

    log.timer_debug1("transforming DF-AO integrals to MO", *cput0)

    return Lchol
