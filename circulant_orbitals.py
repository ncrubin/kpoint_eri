from functools import reduce
from pyscf import gto, scf, ao2mo
import numpy as np
from openfermion import general_basis_change

def main():
    mol = gto.M()
    mol.atom = 'Li 0 0 0; H 0 0 1.6'
    mol.basis = '6-31g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

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

    norbs = mf.mo_coeff.shape[1]
    circulant_orbs = np.zeros((norbs, norbs), dtype=np.complex128)
    for n in range(norbs): # index for circulant orb
        for l in range(norbs): # index for canonical orbital
            circulant_orbs[l, n] = (1/np.sqrt(norbs)) * np.exp(2 * np.pi * 1j * n * l / norbs)

    circulant_oei = general_basis_change(one_electron_integrals, circulant_orbs, (1, 0))
    circulant_tei = general_basis_change(two_electron_integrals, circulant_orbs, (1, 0, 1, 0))
    # circulant_oei = general_basis_change(one_electron_integrals, np.eye(norbs), (1, 0))
    # circulant_tei = general_basis_change(two_electron_integrals, np.eye(norbs), (1, 0, 1, 0))


    assert np.allclose(circulant_tei, circulant_tei.transpose(2, 3, 0, 1))
    assert np.allclose(circulant_tei, circulant_tei.transpose(1, 0, 3, 2).conj())
    assert np.allclose(circulant_tei, circulant_tei.transpose(3, 2, 1, 0).conj())

    w, v = np.linalg.eigh(circulant_tei.transpose(0, 1, 3, 2).reshape((norbs ** 2, norbs ** 2)))
    from utils import modified_cholesky
    L = modified_cholesky(circulant_tei.transpose(0, 1, 3, 2).reshape((norbs**2, norbs**2))).T
    Ltensor = np.zeros((L.shape[1], norbs, norbs), dtype=np.complex128)
    for ll in range(L.shape[1]):
        print(ll, L.shape[1])
        Ltensor[ll] = L[:, [ll]].reshape((norbs, norbs))
        assert np.allclose(Ltensor[ll], Ltensor[ll].conj().T) # this should fail for complex orbitals

main()