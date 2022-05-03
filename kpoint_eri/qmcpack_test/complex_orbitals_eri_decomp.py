from functools import reduce
from itertools import product
from pyscf import gto, scf, ao2mo
import numpy as np
import scipy as sp
from scipy.linalg import expm
from openfermion import general_basis_change
import openfermion as of

from pyscf import lib

def build_true_tei_op_Q(Qindex, df, kpoints, mo_coeff, momentum_map):
    num_kpoints = momentum_map.shape[0]
    true_teo_op = of.FermionOperator()
    nmo = mo_coeff[0].shape[1]
    true_tei_op = of.FermionOperator()
    for kp, ks in product(range(num_kpoints), repeat=2):
        kq = momentum_map[Qindex, kp]
        kr = momentum_map[Qindex, ks]
        kpt_pqrs = [kpoints[ik] for ik in [kp,kq,kr,ks]]
        mos_pqrs = [mo_coeff[ik] for ik in [kp,kq,kr,ks]]
        eri_pqrs = df.ao2mo(mos_pqrs, kpts=kpt_pqrs, compact=False).reshape((nmo,)*4) / num_kpoints
        for p, q, r, s in product(range(nmo), repeat=4):
            # open fermion doesnt like np int.
            P = int(kp * nmo + p)
            Q = int(kq * nmo + q)
            R = int(kr * nmo + r)
            S = int(ks * nmo + s)
            op = ((P, 1), (Q, 0), (R, 1), (S, 0))
            fop = of.FermionOperator(op, coefficient=eri_pqrs[p, q, r, s])
            true_tei_op += fop
    return true_tei_op

def build_rho_Q_n(LQn, Q_index, momentum_map):
    assert len(LQn.shape) == 3
    nmo = LQn.shape[1]
    num_kpoints = LQn.shape[0]
    fermion_op = of.FermionOperator()
    for kp in range(num_kpoints):
        kq = momentum_map[Q_index, kp]
        for p, q in product(range(nmo), repeat=2):
            P = int(kp * nmo + p)
            Q = int(kq * nmo + q)
            oper = ((P, 1), (Q, 0))
            fermion_op += of.FermionOperator(oper, coefficient=LQn[kp, p, q])
    return fermion_op

def get_fermion_op(coeff_tensor, ordering='phys') -> of.FermionOperator:
    r"""Returns an openfermion.FermionOperator from the given coeff_tensor.

    Given A[i, j, k, l] of A = \sum_{ijkl}A[i, j, k, l]i^ j^ k l
    return the FermionOperator A.

    Args:
        coeff_tensor: Coefficients for 4-mode operator

    Returns:
        A FermionOperator object
    """
    if ordering not in ['phys', 'chem']:
        raise ValueError("Invalid input ordering. Must be phys or chem")

    if len(coeff_tensor.shape) == 4:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q, r, s in product(range(nso), repeat=4):
            if ordering == 'phys':
                if p == q or r == s:
                    continue
                op = ((p, 1), (q, 1), (r, 0), (s, 0))
                fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q, r, s])
            elif ordering == 'chem':
                op = ((p, 1), (q, 0), (r, 1), (s, 0))
                fop = of.FermionOperator(op, coefficient=coeff_tensor[p, q, r, s])
            else:
                raise Exception("The impossible has happened!")
            fermion_op += fop
        return fermion_op

    elif len(coeff_tensor.shape) == 2:
        nso = coeff_tensor.shape[0]
        fermion_op = of.FermionOperator()
        for p, q in product(range(nso), repeat=2):
            oper = ((p, 1), (q, 0))
            fop = of.FermionOperator(oper, coefficient=coeff_tensor[p, q])
            fermion_op += fop
        return fermion_op

    else:
        raise ValueError(
            "Arg `coeff_tensor` should have dimension 2 or 4 but has dimension"
            f" {len(coeff_tensor.shape)}.")

def main():
    #############################################
    #
    #   Reconstruct a Fermion Operator from
    #     Cholesky vectors reshaped to one-body
    #     operators. Note this is not the square
    #     of an operator.
    #     It is
    #     \sum_{l}\hat{L_{l}}\hat{L_{l}}^{\dagger}
    #
    #############################################
    from utils import read_qmcpack_hamiltonian
    # too slow in python just sample some randomly
    hamil = read_qmcpack_hamiltonian('hamiltonian.h5')
    chol_vecs = hamil['chol']
    nmo_per_kp = hamil['nmo_pk']
    nchol_per_kp = hamil['nchol_pk']
    qk_to_k2 = hamil['qk_k2']
    nelec = hamil['nelec']
    nk = len(hamil['kpoints'])
    nelec_per_kp = nelec[0] // nk
    kminus = hamil['minus_k']
    LQ = []
    for i, c in enumerate(chol_vecs):
        LQ.append(c.reshape((nk, nmo_per_kp[i], nmo_per_kp[i], nchol_per_kp[i])))
    from pyscf.pbc.df import FFTDF
    from utils import load_from_pyscf_chk
    scf_data = load_from_pyscf_chk('diamond_221.chk')
    df = FFTDF(scf_data['cell'], kpts=scf_data['kpts'])
    kpoints = scf_data['kpts']
    mo_coeff = scf_data['mo_coeff']
    nmo = mo_coeff[0].shape[1] # safe for the moment
    num_kp = len(kpoints)
    nsamp = 3
    kp_idx = np.arange(num_kp)
    # qs = np.random.choice(kp_idx, nsamp, replace=False)
    k1s = np.random.choice(kp_idx, nsamp, replace=False)
    k4s = np.random.choice(kp_idx, nsamp, replace=False)

    ################################################################
    #
    #   Test decomposition for selected value of Q, k1, and k2
    #   (pkp qkq | rkr sks) = (pkp q(Q-kp) | r(Q-ks) s ks)
    #                       = sum_n L[Q,kp,p,q,n]  L[Q,ks,s,r,n].conj()
    #
    ####################################################################
    # Just pick some values arbitrarily for speed Nk^3 is slow.
    iq = 3
    kp = 1
    ks = 2
    kq = qk_to_k2[iq][kp]
    kr = qk_to_k2[iq][ks]
    kpt_pqrs = [kpoints[ik] for ik in [kp,kq,kr,ks]]
    mos_pqrs = [mo_coeff[ik] for ik in [kp,kq,kr,ks]]
    eri_pqrs = df.ao2mo(mos_pqrs, kpts=kpt_pqrs, compact=False).reshape((nmo,)*4) / num_kp
    true_tei_op = get_fermion_op(eri_pqrs, ordering='chem')
    test_tei_op = of.FermionOperator()
    eri_chol = np.einsum('pqn,srn->pqrs', LQ[iq][kp], LQ[iq][ks].conj())
    # Note this is a specific term (not just LL^ so k indices matter)
    for nchol in range(LQ[iq].shape[-1]):
        one_body_cholesky_op_L = get_fermion_op(LQ[iq][kp,:,:,nchol])
        one_body_cholesky_op_R = of.hermitian_conjugated(get_fermion_op(LQ[iq][ks,:,:,nchol]))
        test_op = of.normal_ordered(one_body_cholesky_op_L *
                    one_body_cholesky_op_R -
                    get_fermion_op(np.einsum('pq,sr->pqrs', LQ[iq][kp,:,:,nchol],
                        LQ[iq][ks,:,:,nchol].conj()), ordering='chem'))
        test_tei_op += one_body_cholesky_op_L * one_body_cholesky_op_R
    print((true_tei_op-test_tei_op).induced_norm()) # ~1e-6 with default config value for EQ_TOLERANCE
    ################################################################
    #
    #   Build rho_n(Q) = \sum_{kpn} L_{pq,n}^{Q kp} (p kp)^ (q kp-Q)
    #   Test rho_n(Q) rho_n(Q)^ = TEI
    #   Note (p kp) is compound index so creation / annihilation operators live in
    #   Nk*m space i.e. [0 k0, 1 k0, .., 0 k1, 1 k1, ..] etc. (p kp) = kp * nmo +  p
    #   Assume for the moment nmo_per_kpoint is constant = nmo
    #
    ####################################################################

    # pick some values at random
    iq = 2
    eri_pqrs = df.ao2mo(mos_pqrs, kpts=kpt_pqrs, compact=False).reshape((nmo,)*4) / num_kp
    # (p kp q Q-kp | r Q-ks s ks) for our iq
    true_tei_op = build_true_tei_op_Q(iq, df, kpoints, mo_coeff, qk_to_k2)
    test_tei_op = of.FermionOperator()
    for n in range(LQ[iq].shape[-1]):
        rho_Q_n = build_rho_Q_n(LQ[iq][:,:,:,n], iq, qk_to_k2)
        test_tei_op += rho_Q_n * of.hermitian_conjugated(rho_Q_n)
    print((true_tei_op-test_tei_op).induced_norm()) # ~1e-6 with default config value for EQ_TOLERANCE

    # ################################################################
    # #
    # #   Check if sum of squares decomposition
    # #     See note.
    # #
    # ####################################################################
    # test_tei_op = of.FermionOperator()
    # true_tei_op = get_fermion_op(tei, ordering='chem')
    iq = 2
    # there are roughly 100 cholesky vectors so might take some time.
    for ll in range(LQ[iq].shape[-1]):
        # define U_l and V_l like in the note
        rho_Q_n = build_rho_Q_n(LQ[iq][:,:,:,ll], iq, qk_to_k2)
        U_l = 1j * rho_Q_n
        V_l = of.hermitian_conjugated(rho_Q_n)
        U_ld = of.hermitian_conjugated(U_l)
        V_ld = of.hermitian_conjugated(V_l)

        # this is Eq. A13 in note
        test_tei_op += U_l * V_l
        test_tei_op += V_l * U_l
        test_tei_op -= U_ld * V_ld
        test_tei_op -= V_ld * U_ld
        # Eq. A13 repeated
        ulvl_cholesky_term = U_l * V_l \
                           + V_l * U_l \
                           - U_ld * V_ld \
                           - V_ld * U_ld
        # Eq. A14
        ulvl_sumsquared = 0.5 * ( (U_l + V_l)**2 - (U_l - V_l)**2
                                - (U_ld + V_ld)**2 + (U_ld - V_ld)**2)
        ulvl_c_Vs_ss = (ulvl_sumsquared - ulvl_cholesky_term).induced_norm()
        print(ulvl_c_Vs_ss)
        # assert np.isclose(ulvl_c_Vs_ss, 0, atol=2.0E-6)  # I would guess that adding and subtracting small numbers is the cause of the atol being hgih.

        # Eq. A17
        S_l = U_l + V_l
        S_ld = of.hermitian_conjugated(S_l)
        D_l = U_ld - V_ld
        D_ld = of.hermitian_conjugated(D_l)

        # This is a sum of squares of normal operators that we can DF!
        # Eq. A18
        sldl_term = ((S_l + 1j * S_ld)**2 - (S_l - 1j * S_ld)**2 + (D_l + 1j * D_ld)**2 - (D_l - 1j * D_ld)**2)
        sldl_vs_ulvl = (ulvl_cholesky_term - (1/4) * sldl_term).induced_norm()
        print(sldl_vs_ulvl)
        assert np.isclose(sldl_vs_ulvl, 0)  # I would guess that adding and subtracting small numbers is the cause of the atol being hgih.

    assert np.isclose((true_tei_op - -0.25j * test_tei_op).induced_norm(), 0)



main()
