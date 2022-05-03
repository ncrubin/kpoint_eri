import numpy as np
import time

def test_kp_eri():
    h2 = np.load('h2_full.npz')['v2']

def generate_hamiltonian(nmo, cplx=False, sym=8):
    eri = np.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    assert cplx == (sym == 4)
    if cplx:
        eri = eri + 1j*np.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    # Restore symmetry to the integrals.
    if sym >= 4:
        # (ik|jl) = (jl|ik)
        # (ik|jl) = (ki|lj)*
        eri = eri + eri.transpose(2,3,0,1)
        eri = eri + eri.transpose(3,2,1,0).conj()
    if sym == 8:
        eri = eri + eri.transpose(1,0,2,3)
    return eri

def modified_cholesky(M, tol=1e-6, verbose=True, cmax=20):
    """Modified cholesky decomposition of matrix.

    See, e.g. [Motta17]_

    Parameters
    ----------
    M : :class:`np.ndarray`
        Positive semi-definite, symmetric matrix.
    tol : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    assert len(M.shape) == 2
    delta = np.copy(M.diagonal())
    nchol_max = int(cmax*M.shape[0]**0.5)
    # index of largest diagonal element of residual matrix.
    nu = np.argmax(np.abs(delta))
    delta_max = delta[nu]
    if verbose:
        print ("# max number of cholesky vectors = %d"%nchol_max)
        print ("# iteration %d: delta_max = %f"%(0, delta_max.real))
    # Store for current approximation to input matrix.
    Mapprox = np.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = np.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = np.copy(M[:,nu])/delta_max**0.5
    while abs(delta_max) > tol:
        # Update cholesky vector
        start = time.time()
        Mapprox += chol_vecs[nchol]*chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        nchol += 1
        Munu0 = np.dot(chol_vecs[:nchol,nu].conj(), chol_vecs[:nchol,:])
        chol_vecs[nchol] = (M[:,nu] - Munu0) / (delta_max)**0.5
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %d: delta_max = %13.8e: time = %13.8e"%info)

    return np.array(chol_vecs[:nchol])

def test_cholesky():
    nmo = 10
    eri = generate_hamiltonian(10, True, sym=4)
    eri = eri.transpose((0,1,3,2))
    eri = eri.reshape((nmo*nmo, nmo*nmo))
    eri = eri @ eri.conj().T # positive semi definite.
    chol = modified_cholesky(eri, tol=1e-5, verbose=True, cmax=30)

def test_kp_eri():
    eri = np.load('h2_full.npz')['v2'] # full four index tensor saved from bloch_orbital_tei (~250 MB so didn't add)
    nmo = eri.shape[0]
    # Form Hermitian matrix M_{pq,sr} = (pq|rs)
    eri_herm = eri.transpose((0,1,3,2))
    eri_herm = eri_herm.reshape((nmo*nmo, nmo*nmo))
    chol = modified_cholesky(eri_herm, tol=1e-5, verbose=False, cmax=30)
    print("rank: {:d} vs dim {:d}".format(len(chol), nmo**2))
    print("saving: {:f} %".format(100*(1-len(chol)/nmo**2)))
    print("|max error|: {:13.8e}".format(
            np.abs(
                np.max(
                    np.einsum('PI,PJ->IJ', chol, chol.conj(), optimize=True)
                    -
                    eri_herm
                    )
                )
            )
        )
    for L in chol[:10]:
        # These should not be Hermitian in general
        LP = L.reshape((nmo,nmo))
        assert np.linalg.norm(L-L.conj().T) > 0.0
    # Look at Mario's SVD
    chol_mat = np.array(chol).T.copy()
    U, sigma, Vh = np.linalg.svd(chol_mat)
    nchol = chol_mat.shape[1]
    eri_svd = np.einsum('Iu,u,Ju->IJ', U[:,:nchol], sigma**2.0, U[:,:nchol].conj(), optimize=True)
    print("|max error (svd)|: {:13.8e}".format(
            np.max(
                np.abs(
                    eri_svd - eri_herm
                    )
                )
            )
        )

if __name__ == '__main__':
    # test_cholesky()
    test_kp_eri()
