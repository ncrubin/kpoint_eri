#!/usr/bin/env python
# Copyright 2017-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#
# Modified to overwrite integral generation step to force incore and avoid
# ao2mo7d type calls + simplified slightly.
from functools import reduce
import numpy as np

import copy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import (
    get_nocc,
    padded_mo_coeff,
    padding_k_idx,
)  # noqa
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
    def __init__(self, cc, mo_coeff=None, method="incore", eri_helper=None):
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
        self.fock = np.asarray(
            [
                reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                for k, mo in enumerate(mo_coeff)
            ]
        )
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
            self.mo_energy = [
                _adjust_occ(mo_e, nocc, -madelung)
                for k, mo_e in enumerate(self.mo_energy)
            ]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt) - 1]
        # if gap < 1e-5:
        # logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
        # 'May cause issues in convergence.', gap)

        # mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        orbv = np.asarray(mo_coeff[:, :, nocc:], order="C")

        # log.info('using incore ERI storage')
        self.oooo = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=dtype)
        self.ooov = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype=dtype)
        self.oovv = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=dtype)
        self.ovov = np.empty((nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype=dtype)
        self.voov = np.empty((nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype=dtype)
        self.vovv = np.empty((nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), dtype=dtype)
        self.vvvv = np.empty((nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), dtype=dtype)
        # self.vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

        for (ikp, ikq, ikr) in khelper.symm_map.keys():
            iks = kconserv[ikp, ikq, ikr]
            # eri_kpt = ((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
            # (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
            kpts = [ikp, ikq, ikr, iks]
            eri_kpt = eri_helper.get_eri(kpts)
            if dtype == float:
                eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(
                    0, 2, 1, 3
                )
                self.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                self.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
                self.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                self.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                self.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                self.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
                self.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

        self.dtype = dtype

def update_eris(cc, eris, eri_helper, inplace=True):
    """Update coupled cluster eris object with approximate integrals defined by eri_helper. 

    Arguments:
        cc: pyscf PBC KRCCSD object. On output the eris attribute will be
        modified by eri_helper.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
    """
    log = logger.Logger(cc.stdout, cc.verbose)
    kconserv = cc.khelper.kconserv
    khelper = cc.khelper
    nocc = cc.nocc
    nkpts = cc.nkpts
    dtype = cc.mo_coeff[0].dtype
    if inplace:
        log.info(f"Modifying inplace coupled cluster _ERIS object using {eri_helper.__class__}.")
        out_eris = eris
    else:
        log.info(f"Rebuilding coupled cluster _ERIS object using {eri_helper.__class__}.")
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