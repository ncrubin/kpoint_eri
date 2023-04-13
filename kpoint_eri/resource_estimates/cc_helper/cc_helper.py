"""Utilities for overwriting CCSD pbc eris with integral factorizations."""
import copy

from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import loop_kkk
from pyscf.pbc.cc.kccsd_uhf import _make_df_eris


def build_approximate_eris(cc, eris, eri_helper, inplace=True):
    """Update coupled cluster eris object with approximate integrals defined by eri_helper.

    Arguments:
        cc: pyscf PBC KRCCSD object.
        eris: pyscf _ERIS object.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
        inplace: If true we overwrite the input eris integrals with those
            constructed from eri_helper.

    Returns:
        eris: pyscf _ERIS object updated to hold approximate eris defined by
            eri_helper. 
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
        eri_kpt = eri_helper.get_eri(kpts) / nkpts
        if dtype == float:
            eri_kpt = eri_kpt.real
        eri_kpt = eri_kpt
        for kp, kq, kr in khelper.symm_map[(ikp, ikq, ikr)]:
            eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(
                0, 2, 1, 3
            )
            out_eris.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc]
            out_eris.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:]
            out_eris.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:]
            out_eris.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:]
            out_eris.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:]
            out_eris.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:]
            out_eris.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:]
    return out_eris


def build_approximate_eris_rohf(cc, eris, eri_helper, inplace=True):
    """Update coupled cluster eris object with approximate integrals defined by eri_helper.

    Arguments:
        cc: pyscf PBC KUCCSD object. Only ROHF integrals are supported.
        eris: pyscf _ERIS object.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
        inplace: If true we overwrite the input eris integrals with those
            constructed from eri_helper.

    Returns:
        eris: pyscf _ERIS object updated to hold approximate eris defined by
            eri_helper. 
    """
    log = logger.Logger(cc.stdout, cc.verbose)
    kconserv = cc.khelper.kconserv
    khelper = cc.khelper
    nocca, noccb = cc.nocc
    nkpts = cc.nkpts
    if inplace:
        log.info(
            f"Modifying inplace coupled cluster _ERIS object using {eri_helper.__class__}."
        )
        out_eris = eris
    else:
        log.info(
            f"Rebuilding coupled cluster _ERIS object using {eri_helper.__class__}."
        )
        out_eris = _make_df_eris(cc) 
    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.oooo[kp, kq, kr] = tmp[:nocca, :nocca, :nocca, :nocca]
        out_eris.ooov[kp, kq, kr] = tmp[:nocca, :nocca, :nocca, nocca:]
        out_eris.oovv[kp, kq, kr] = tmp[:nocca, :nocca, nocca:, nocca:]
        out_eris.ovov[kp, kq, kr] = tmp[:nocca, nocca:, :nocca, nocca:]
        out_eris.voov[kq, kp, ks] = (
            tmp[:nocca, nocca:, nocca:, :nocca].conj().transpose(1, 0, 3, 2)
        )
        out_eris.vovv[kq, kp, ks] = (
            tmp[:nocca, nocca:, nocca:, nocca:].conj().transpose(1, 0, 3, 2)
        )

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.OOOO[kp, kq, kr] = tmp[:noccb, :noccb, :noccb, :noccb]
        out_eris.OOOV[kp, kq, kr] = tmp[:noccb, :noccb, :noccb, noccb:]
        out_eris.OOVV[kp, kq, kr] = tmp[:noccb, :noccb, noccb:, noccb:]
        out_eris.OVOV[kp, kq, kr] = tmp[:noccb, noccb:, :noccb, noccb:]
        out_eris.VOOV[kq, kp, ks] = (
            tmp[:noccb, noccb:, noccb:, :noccb].conj().transpose(1, 0, 3, 2)
        )
        out_eris.VOVV[kq, kp, ks] = (
            tmp[:noccb, noccb:, noccb:, noccb:].conj().transpose(1, 0, 3, 2)
        )

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.ooOO[kp, kq, kr] = tmp[:nocca, :nocca, :noccb, :noccb]
        out_eris.ooOV[kp, kq, kr] = tmp[:nocca, :nocca, :noccb, noccb:]
        out_eris.ooVV[kp, kq, kr] = tmp[:nocca, :nocca, noccb:, noccb:]
        out_eris.ovOV[kp, kq, kr] = tmp[:nocca, nocca:, :noccb, noccb:]
        out_eris.voOV[kq, kp, ks] = (
            tmp[:nocca, nocca:, noccb:, :noccb].conj().transpose(1, 0, 3, 2)
        )
        out_eris.voVV[kq, kp, ks] = (
            tmp[:nocca, nocca:, noccb:, noccb:].conj().transpose(1, 0, 3, 2)
        )

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        # out_eris.OOoo[kp,kq,kr] = tmp[:noccb,:noccb,:nocca,:nocca]
        out_eris.OOov[kp, kq, kr] = tmp[:noccb, :noccb, :nocca, nocca:]
        out_eris.OOvv[kp, kq, kr] = tmp[:noccb, :noccb, nocca:, nocca:]
        out_eris.OVov[kp, kq, kr] = tmp[:noccb, noccb:, :nocca, nocca:]
        out_eris.VOov[kq, kp, ks] = (
            tmp[:noccb, noccb:, nocca:, :nocca].conj().transpose(1, 0, 3, 2)
        )
        out_eris.VOvv[kq, kp, ks] = (
            tmp[:noccb, noccb:, nocca:, nocca:].conj().transpose(1, 0, 3, 2)
        )
    # Assuming spin-free integrals (RHF/ROHF)
    for ki in range(nkpts):
        for kj in range(nkpts):
            out_eris.Lpv[ki, kj] = eri_helper.chol[ki, kj][:, :, nocca:]
            out_eris.LPV[ki, kj] = eri_helper.chol[ki, kj][:, :, noccb:]

    return out_eris
