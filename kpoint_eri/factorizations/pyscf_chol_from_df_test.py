import itertools
import numpy as np

from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints 

def test_pyscf_chol_from_df():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-cc-dzvp"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 4
    cell.build()

    name_prefix = ""
    basname = cell.basis
    pp_name = cell.pseudo

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    # use regular density fitting for compatibility with pyscf pip release
    # uncomment for rs density fitting:
    # mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.init_guess = "chkfile"
    Escf = mf.kernel()

    mymp = mp.KMP2(mf)
    nmo = mymp.nmo
    nocc = mymp.nocc
    nvir = nmo - nocc
    Luv = cholesky_from_df_ints(mymp)

    # 1. Test that the DF integrals give the correct SCF energy (oo block)
    mf.exxdiv = None  # exclude ewald exchange correction
    Eref = mf.energy_elec()[1]
    Eout = 0.0j
    for ik, jk in itertools.product(range(nkpts), repeat=2):
        Lii = Luv[ik, ik][:, :nocc, :nocc]
        Ljj = Luv[jk, jk][:, :nocc, :nocc]
        Lij = Luv[ik, jk][:, :nocc, :nocc]
        Lji = Luv[jk, ik][:, :nocc, :nocc]
        oooo_d = np.einsum("Lij,Lkl->ijkl", Lii, Ljj) / nkpts
        oooo_x = np.einsum("Lij,Lkl->ijkl", Lij, Lji) / nkpts
        Eout += 2.0 * np.einsum("iijj->", oooo_d)
        Eout -= np.einsum("ijji->", oooo_x)
    assert abs(Eout / nkpts - Eref) < 1e-12

    # 2. Test that the DF integrals agree with those from MP2 (ov block)
    from pyscf.pbc.mp.kmp2 import _init_mp_df_eris

    Ltest = _init_mp_df_eris(mymp)
    for ik, jk in itertools.product(range(nkpts), repeat=2):
        assert np.allclose(Luv[ik, jk][:, :nocc, nocc:], Ltest[ik, jk], atol=1e-12)

    # 3. Test that the DF integrals have correct vvvv block (vv)
    Ivvvv = np.zeros((nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), dtype=np.complex128)
    for ik, jk, kk in itertools.product(range(nkpts), repeat=3):
        lk = mymp.khelper.kconserv[ik, jk, kk]
        Lij = Luv[ik, jk][:, nocc:, nocc:]
        Lkl = Luv[kk, lk][:, nocc:, nocc:]
        Imo = np.einsum("Lij,Lkl->ijkl", Lij, Lkl)
        Ivvvv[ik, jk, kk] = Imo / nkpts

    mycc = cc.KRCCSD(mf)
    eris = mycc.ao2mo()
    assert np.allclose(eris.vvvv, Ivvvv.transpose(0, 2, 1, 3, 5, 4, 6), atol=1e-12)