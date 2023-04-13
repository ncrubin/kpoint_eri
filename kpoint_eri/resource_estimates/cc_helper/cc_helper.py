"""Utilities for overwriting CCSD pbc eris with integral factorizations."""
import copy

from pyscf.pbc import cc
from pyscf.lib import logger


def build_approximate_eris(cc, eris, eri_helper, inplace=True):
    """Update coupled cluster eris object with approximate integrals defined by eri_helper.

    Arguments:
        cc: pyscf PBC KRCCSD object.
        eris: pyscf _ERIS object.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
        inplace: If true we overwrite the input eris integrals with those
            constructed from eri_helper.
    """
    log = logger.Logger(cc.stdout, cc.verbose)
    kconserv = cc.khelper.kconserv
    khelper = cc.khelper
    nocc = cc.nocc
    nkpts = cc.nkpts
    dtype = cc.mo_coeff[0].dtype
    if inplace:
        log.info(
            f"Modifying inplace coupled cluster _ERIS object using {eri_helper.__class__}."
        )
        out_eris = eris
    else:
        log.info(
            f"Rebuilding coupled cluster _ERIS object using {eri_helper.__class__}."
        )
        out_eris = copy.deepcopy(eris)
    for ikp, ikq, ikr in khelper.symm_map.keys():
        iks = kconserv[ikp, ikq, ikr]
        kpts = [ikp, ikq, ikr, iks]
        eri_kpt = eri_helper.get_eri(kpts)
        if dtype == float:
            eri_kpt = eri_kpt.real
        eri_kpt = eri_kpt
        for kp, kq, kr in khelper.symm_map[(ikp, ikq, ikr)]:
            eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(
                0, 2, 1, 3
            )
            out_eris.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
            out_eris.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
            out_eris.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
            out_eris.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
            out_eris.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
            out_eris.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
            out_eris.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts
    return out_eris


def compute_emp2_approx(exact_cc, helper) -> float:
    """Compute approximate MP2 energy using integral helper

    Args:
        exact_cc: Reference pbc KRCCSD object.
        helper: integral helper.

    Returns:
        emp2: MP2 energy
    """
    if hasattr(exact_cc, "eris"):
        eris_exact = exact_cc.eris
    else:
        eris_exact = exact_cc.ao2mo()

    approx_cc = cc.KRCCSD(exact_cc._scf)
    approx_eris = update_eris(cc, eris_exact, helper)
    emp2, _, _ = approx_cc.init_amps(approx_eris)
    emp2 += exact_cc._scf.e_tot
    return emp2
