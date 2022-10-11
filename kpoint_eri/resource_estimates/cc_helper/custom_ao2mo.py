from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf.cc import uccsd
from pyscf.pbc.mp.kmp2 import (get_nocc,
                               padded_mo_coeff, padding_k_idx)  # noqa
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import __config__

#######################################
#
# _ERIS.
#
# Note the two electron integrals are stored in different orders from
# kccsd_uhf._ERIS.  Integrals (ab|cd) are stored as [ka,kc,kb,a,c,b,d] here
# while the order is [ka,kb,kc,a,b,c,d] in kccsd_uhf._ERIS
#
# TODO: use the same convention as kccsd_uhf
#
class _ERIS:  # (pyscf.cc.ccsd._ChemistsERIs):
    def __init__(self, cc, mo_coeff=None, method='incore', eri_helper=None):
        from pyscf.pbc import df
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        # log = logger.Logger(cc.stdout, cc.verbose)
        # cput0 = (logger.process_clock(), logger.perf_counter())
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        # if any(nocc != np.count_nonzero(cc._scf.mo_occ[k]>0)
        #       for k in range(nkpts)):
        #    raise NotImplementedError('Different occupancies found for different k-points')

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        # try:
        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)
        # except IndexError:
            # mo_coeff = self.mo_coeff

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        exxdiv = cc._scf.exxdiv if cc.keep_exxdiv else None
        with lib.temporary_env(cc._scf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            vhf = cc._scf.get_veff(cell, dm)
        fockao = cc._scf.get_hcore() + vhf
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
        self.e_hf = cc._scf.energy_tot(dm=dm, vhf=vhf)
        # print("ehf : ", self.e_hf/2)

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]

        if not cc.keep_exxdiv:
            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
            # Add HFX correction in the self.mo_energy to improve convergence in
            # CCSD iteration. It is useful for the 2D systems since their occupied and
            # the virtual orbital energies may overlap which may lead to numerical
            # issue in the CCSD iterations.
            # FIXME: Whether to add this correction for other exxdiv treatments?
            # Without the correction, MP2 energy may be largely off the correct value.
            madelung = tools.madelung(cell, kpts)
            self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                              for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        # if gap < 1e-5:
            # logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            # 'May cause issues in convergence.', gap)

        # mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')

        # log.info('using incore ERI storage')
        self.oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
        self.ooov = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
        self.oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
        self.ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
        self.voov = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
        self.vovv = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
        self.vvvv = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)
        # self.vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

        for (ikp,ikq,ikr) in khelper.symm_map.keys():
            iks = kconserv[ikp,ikq,ikr]
            # eri_kpt = ((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                             # (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
            kpts = [ikp, ikq, ikr, iks]
            eri_kpt = eri_helper.get_eri(kpts)
            if dtype == np.float: eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                self.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                self.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
                self.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                self.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                self.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                self.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
                self.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

        self.dtype = dtype


def _custom_make_df_eris(cc, eri_helper, mo_coeff=None):
    from pyscf.pbc.df import df
    from pyscf.ao2mo import _ao2mo
    cell = cc._scf.cell
    if cell.dimension == 2:
        raise NotImplementedError

    eris = uccsd._ChemistsERIs()
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
    from pyscf.pbc.mp.kump2 import padded_mo_coeff
    mo_coeff = padded_mo_coeff(cc, mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = cc.nocc
    print(eris.nocc)
    thisdf = cc._scf.with_df

    kpts = cc.kpts
    nkpts = cc.nkpts
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    #if getattr(thisdf, 'auxcell', None):
    #    naux = thisdf.auxcell.nao_nr()
    #else:
    #    naux = thisdf.get_naoaux()
    nao = cell.nao_nr()
    mo_kpts_a, mo_kpts_b = eris.mo_coeff

    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_kpts_a)

    eris.feri = feri = lib.H5TmpFile()

    eris.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocca,nocca,nocca,nocca), dtype)
    eris.ooov = feri.create_dataset('ooov', (nkpts,nkpts,nkpts,nocca,nocca,nocca,nvira), dtype)
    eris.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira), dtype)
    eris.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocca,nvira,nocca,nvira), dtype)
    eris.voov = feri.create_dataset('voov', (nkpts,nkpts,nkpts,nvira,nocca,nocca,nvira), dtype)
    eris.vovv = feri.create_dataset('vovv', (nkpts,nkpts,nkpts,nvira,nocca,nvira,nvira), dtype)
    eris.vvvv = None

    eris.OOOO = feri.create_dataset('OOOO', (nkpts,nkpts,nkpts,noccb,noccb,noccb,noccb), dtype)
    eris.OOOV = feri.create_dataset('OOOV', (nkpts,nkpts,nkpts,noccb,noccb,noccb,nvirb), dtype)
    eris.OOVV = feri.create_dataset('OOVV', (nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb), dtype)
    eris.OVOV = feri.create_dataset('OVOV', (nkpts,nkpts,nkpts,noccb,nvirb,noccb,nvirb), dtype)
    eris.VOOV = feri.create_dataset('VOOV', (nkpts,nkpts,nkpts,nvirb,noccb,noccb,nvirb), dtype)
    eris.VOVV = feri.create_dataset('VOVV', (nkpts,nkpts,nkpts,nvirb,noccb,nvirb,nvirb), dtype)
    eris.VVVV = None

    eris.ooOO = feri.create_dataset('ooOO', (nkpts,nkpts,nkpts,nocca,nocca,noccb,noccb), dtype)
    eris.ooOV = feri.create_dataset('ooOV', (nkpts,nkpts,nkpts,nocca,nocca,noccb,nvirb), dtype)
    eris.ooVV = feri.create_dataset('ooVV', (nkpts,nkpts,nkpts,nocca,nocca,nvirb,nvirb), dtype)
    eris.ovOV = feri.create_dataset('ovOV', (nkpts,nkpts,nkpts,nocca,nvira,noccb,nvirb), dtype)
    eris.voOV = feri.create_dataset('voOV', (nkpts,nkpts,nkpts,nvira,nocca,noccb,nvirb), dtype)
    eris.voVV = feri.create_dataset('voVV', (nkpts,nkpts,nkpts,nvira,nocca,nvirb,nvirb), dtype)
    eris.vvVV = None

    eris.OOoo = None
    eris.OOov = feri.create_dataset('OOov', (nkpts,nkpts,nkpts,noccb,noccb,nocca,nvira), dtype)
    eris.OOvv = feri.create_dataset('OOvv', (nkpts,nkpts,nkpts,noccb,noccb,nvira,nvira), dtype)
    eris.OVov = feri.create_dataset('OVov', (nkpts,nkpts,nkpts,noccb,nvirb,nocca,nvira), dtype)
    eris.VOov = feri.create_dataset('VOov', (nkpts,nkpts,nkpts,nvirb,noccb,nocca,nvira), dtype)
    eris.VOvv = feri.create_dataset('VOvv', (nkpts,nkpts,nkpts,nvirb,noccb,nvira,nvira), dtype)
    eris.VVvv = None

    _custom_kuccsd_eris_common_(cc, eris, eri_helper)

    # Assuming spin-free integrals (RHF/ROHF)
    eris.Lpv = np.empty((nkpts, nkpts), dtype=object)
    eris.LPV = np.empty((nkpts, nkpts), dtype=object)
    for ki in range(nkpts):
        for kj in range(nkpts):
            eris.Lpv[ki,kj] = eri_helper.chol[ki,kj][:,:,nocca:]
            eris.LPV[ki,kj] = eri_helper.chol[ki,kj][:,:,noccb:]

    return eris

def _custom_kuccsd_eris_common_(cc, eris, eri_helper):
    from pyscf.pbc import tools
    from pyscf.pbc.cc.ccsd import _adjust_occ
    #if not (cc.frozen is None or cc.frozen == 0):
    #    raise NotImplementedError('cc.frozen = %s' % str(cc.frozen))

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(cc)
    cell = cc._scf.cell

    kpts = cc.kpts
    nkpts = cc.nkpts
    mo_coeff = eris.mo_coeff
    nocca, noccb = eris.nocc
    mo_a, mo_b = mo_coeff

    # Re-make our fock MO matrix elements from density and fock AO
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    hcore = cc._scf.get_hcore()
    with lib.temporary_env(cc._scf, exxdiv=None):
        vhf = cc._scf.get_veff(cell, dm)
    focka = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[0][k], mo))
             for k, mo in enumerate(mo_a)]
    fockb = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[1][k], mo))
             for k, mo in enumerate(mo_b)]
    eris.fock = (np.asarray(focka), np.asarray(fockb))

    madelung = tools.madelung(cell, kpts)
    mo_ea = [focka[k].diagonal().real for k in range(nkpts)]
    mo_eb = [fockb[k].diagonal().real for k in range(nkpts)]
    mo_ea = [_adjust_occ(e, nocca, -madelung) for e in mo_ea]
    mo_eb = [_adjust_occ(e, noccb, -madelung) for e in mo_eb]
    eris.mo_energy = (mo_ea, mo_eb)

    # The momentum conservation array
    kconserv = cc.khelper.kconserv

    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts)
        eris.oooo[kp,kq,kr] = tmp[:nocca,:nocca,:nocca,:nocca]
        eris.ooov[kp,kq,kr] = tmp[:nocca,:nocca,:nocca,nocca:]
        eris.oovv[kp,kq,kr] = tmp[:nocca,:nocca,nocca:,nocca:]
        eris.ovov[kp,kq,kr] = tmp[:nocca,nocca:,:nocca,nocca:]
        eris.voov[kq,kp,ks] = tmp[:nocca,nocca:,nocca:,:nocca].conj().transpose(1,0,3,2)
        eris.vovv[kq,kp,ks] = tmp[:nocca,nocca:,nocca:,nocca:].conj().transpose(1,0,3,2)

    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts)
        eris.OOOO[kp,kq,kr] = tmp[:noccb,:noccb,:noccb,:noccb]
        eris.OOOV[kp,kq,kr] = tmp[:noccb,:noccb,:noccb,noccb:]
        eris.OOVV[kp,kq,kr] = tmp[:noccb,:noccb,noccb:,noccb:]
        eris.OVOV[kp,kq,kr] = tmp[:noccb,noccb:,:noccb,noccb:]
        eris.VOOV[kq,kp,ks] = tmp[:noccb,noccb:,noccb:,:noccb].conj().transpose(1,0,3,2)
        eris.VOVV[kq,kp,ks] = tmp[:noccb,noccb:,noccb:,noccb:].conj().transpose(1,0,3,2)

    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts)
        eris.ooOO[kp,kq,kr] = tmp[:nocca,:nocca,:noccb,:noccb]
        eris.ooOV[kp,kq,kr] = tmp[:nocca,:nocca,:noccb,noccb:]
        eris.ooVV[kp,kq,kr] = tmp[:nocca,:nocca,noccb:,noccb:]
        eris.ovOV[kp,kq,kr] = tmp[:nocca,nocca:,:noccb,noccb:]
        eris.voOV[kq,kp,ks] = tmp[:nocca,nocca:,noccb:,:noccb].conj().transpose(1,0,3,2)
        eris.voVV[kq,kp,ks] = tmp[:nocca,nocca:,noccb:,noccb:].conj().transpose(1,0,3,2)

    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts)
        # eris.OOoo[kp,kq,kr] = tmp[:noccb,:noccb,:nocca,:nocca]
        eris.OOov[kp,kq,kr] = tmp[:noccb,:noccb,:nocca,nocca:]
        eris.OOvv[kp,kq,kr] = tmp[:noccb,:noccb,nocca:,nocca:]
        eris.OVov[kp,kq,kr] = tmp[:noccb,noccb:,:nocca,nocca:]
        eris.VOov[kq,kp,ks] = tmp[:noccb,noccb:,nocca:,:nocca].conj().transpose(1,0,3,2)
        eris.VOvv[kq,kp,ks] = tmp[:noccb,noccb:,nocca:,nocca:].conj().transpose(1,0,3,2)

    log.timer('CCSD integral transformation', *cput0)
    return eris
