from functools import reduce
from itertools import product
from pyscf import gto, scf, ao2mo
import numpy as np
import scipy as sp
from scipy.linalg import expm
from openfermion import general_basis_change
import openfermion as of


def get_fermion_op(coeff_tensor, ordering='phys') -> of.FermionOperator:
    r"""Returns an openfermion.FermionOperator from the given coeff_tensor.

    Given A[i, j, k, l] of A = \sum_{ijkl}A[i, j, k, l]i^ j^ k^ l
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
    #    Get molecular system and solve SCF
    #
    #############################################
    mol = gto.M()
    mol.atom = 'Li 0 0 0; H 0 0 1.6'
    mol.basis = '6-31g'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    num_alpha, num_beta = int(mf.mol.nelectron / 2), int(mf.mol.nelectron / 2)
    nocc = num_alpha + num_beta
    nvirt = mf.mol.nao - nocc
    norbs = mf.mo_coeff.shape[1]

    #############################################
    #
    #    Get molecular orbital integrals
    #
    #############################################
    n_orbitals = mf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (mf.mo_coeff.T,
                                              mf.get_hcore(),
                                              mf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(mol, mf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)

    #############################################
    #
    #    Generate circulant orbital rotation
    #       or generate random orbital rotation
    #
    #############################################
    # random full space basis rotation
    np.random.seed(53)
    X, Y = np.random.randn(norbs**2).reshape((norbs, norbs)), np.random.randn(norbs**2).reshape((norbs, norbs))
    X = X - X.T
    Y = 1j * (Y + Y.T)
    assert np.allclose(X, -X.conj().T)
    assert np.allclose(Y, -Y.conj().T)
    basis_rotation = expm(X + Y)
    # create circulant orbital
    # circulant_orbs = np.zeros((norbs, norbs), dtype=np.complex128)
    # for n in range(norbs): # index for circulant orb
    #     for l in range(norbs): # index for canonical orbital
    #         circulant_orbs[l, n] = (1/np.sqrt(norbs)) * np.exp(2 * np.pi * 1j * n * l / norbs)
    # basis_rotation = circulant_orbs


    #############################################
    #
    #    Rotate orbitals and check 4-fold sym.
    #      is preserved
    #
    #############################################
    oei = general_basis_change(one_electron_integrals, basis_rotation, (1, 0))
    tei = general_basis_change(two_electron_integrals, basis_rotation, (1, 0, 1, 0))
    assert np.allclose(tei, tei.transpose(2, 3, 0, 1))
    assert np.allclose(tei, tei.transpose(1, 0, 3, 2).conj())
    assert np.allclose(tei, tei.transpose(3, 2, 1, 0).conj())
    # check Hartree-Fock energy


    #############################################
    #
    #   Build Cholesky vectors and check if
    #    they can be used to reconstruct TEI
    #
    #############################################
    w, v = np.linalg.eigh(tei.transpose(0, 1, 3, 2).reshape((norbs ** 2, norbs ** 2)))
    assert np.alltrue(w >= -1.0E-14)  # assert positive definite two-electron integrals
    ##################
    #
    # Cholesky 2 ways
    #
    ##################
    ## from iterative
    from utils import modified_cholesky
    L = modified_cholesky(tei.transpose(0, 1, 3, 2).reshape((norbs**2, norbs**2)), tol=1.0E-10).T  # set tolerance very low for comparison tests later on
    ## from eigendecomp
    # pos_w = np.where(w > 1.0E-10)[0][::-1]
    # L = np.zeros((norbs**2, len(pos_w)), dtype=tei.dtype)
    # for idx, ll in enumerate(pos_w):
    #     L[:, [idx]] = np.sqrt(w[ll]) * v[:, [ll]]

    tei_mat = tei.transpose(0, 1, 3, 2).reshape((norbs**2, norbs**2))
    assert np.allclose(L @ L.conj().T, tei_mat)  # 1.0E-6 is the cholesky vector error
    Ltensor = np.zeros((L.shape[1], norbs, norbs), dtype=np.complex128)
    for ll in range(L.shape[1]):
        Ltensor[ll] = L[:, [ll]].reshape((norbs, norbs))
    Ltensor_test = L.T.reshape((-1, norbs, norbs))  # because reshape is row-major iteration
    assert np.allclose(Ltensor_test, Ltensor)
    assert np.allclose(tei, np.einsum('Lpq,Lsr->pqrs', Ltensor, Ltensor.conj()))  # atol=1.0E-6 because  modified_cholesky construction precision

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
    test_tei_op = of.FermionOperator()
    true_tei_op = get_fermion_op(tei, ordering='chem')
    for ll in range(Ltensor.shape[0]):
        one_body_cholesky_op = get_fermion_op(Ltensor[ll])
        test_tei_op += one_body_cholesky_op * of.hermitian_conjugated(one_body_cholesky_op)
        test_op = of.normal_ordered(one_body_cholesky_op * of.hermitian_conjugated(one_body_cholesky_op) - get_fermion_op(np.einsum('pq,sr->pqrs', Ltensor[ll], Ltensor[ll].conj()), ordering='chem'))
        assert np.isclose(test_op.induced_norm(), 0)
    assert np.isclose((true_tei_op - test_tei_op).induced_norm(), 0)

    ################################################################
    #
    #   Check if
    #    TEI = 0.5 (\sum_{l}\hat{L_{l}}\hat{L_{l}}^{\dagger} +
    #               \hat{L_{l}}^{\dagger}\hat{L_{l}})
    #
    ####################################################################
    test_tei_op = of.FermionOperator()
    true_tei_op = get_fermion_op(tei, ordering='chem')
    for ll in range(Ltensor.shape[0]):
        one_body_cholesky_op = get_fermion_op(Ltensor[ll])
        test_tei_op += 0.5 * one_body_cholesky_op * of.hermitian_conjugated(one_body_cholesky_op)
        test_tei_op += 0.5 * of.hermitian_conjugated(one_body_cholesky_op) * one_body_cholesky_op
    assert np.isclose((true_tei_op - test_tei_op).induced_norm(), 0)



    ################################################################
    #
    #   Check if sum of squares decomposition
    #     See note.
    #
    ####################################################################
    test_tei_op = of.FermionOperator()
    true_tei_op = get_fermion_op(tei, ordering='chem')
    for ll in range(Ltensor.shape[0]):
        # define U_l and V_l like in the note
        U_l = 1j * get_fermion_op(Ltensor[ll])
        V_l = of.hermitian_conjugated(get_fermion_op(Ltensor[ll]))
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
        # print(ulvl_c_Vs_ss)
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