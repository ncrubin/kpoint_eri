import numpy as np
from scipy.linalg import schur
import openfermion as of
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.scf.chkfile import load_scf
from pyscf.pbc import gto
from itertools import product
from pyscf.pbc.lib.kpts_helper import member
from collections import defaultdict
import numpy
import time


def modified_cholesky(M, tol=1e-6, verbose=True, cmax=20):
    """Modified cholesky decomposition of matrix.
    See, e.g. [Motta17]_
    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Positive semi-definite, symmetric matrix.
    tol : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    assert len(M.shape) == 2
    delta = numpy.copy(M.diagonal())
    nchol_max = int(cmax*M.shape[0]**0.5)
    # index of largest diagonal element of residual matrix.
    nu = numpy.argmax(numpy.abs(delta))
    delta_max = delta[nu]
    if verbose:
        print ("# max number of cholesky vectors = %d"%nchol_max)
        print ("# iteration %d: delta_max = %f"%(0, delta_max.real))
    # Store for current approximation to input matrix.
    Mapprox = numpy.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = numpy.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = numpy.copy(M[:,nu])/delta_max**0.5
    while abs(delta_max) > tol:
        # Update cholesky vector
        start = time.time()
        Mapprox += chol_vecs[nchol]*chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        nchol += 1
        Munu0 = numpy.dot(chol_vecs[:nchol,nu].conj(), chol_vecs[:nchol,:])
        chol_vecs[nchol] = (M[:,nu] - Munu0) / (delta_max)**0.5
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %d: delta_max = %13.8e: time = %13.8e"%info)

    return numpy.array(chol_vecs[:nchol])


def main():
    cell, scf_dict = load_scf('diamond_222')
    kpts = scf_dict['kpts']
    khelper = kpts_helper.KptsHelper(cell, kpts)
    kconserv = khelper.kconserv
    nkpts = len(kpts)
    h1 = np.load('h1_222.npy')  # [k, nmo, nmo]
    nmo = h1.shape[-1]
    h2 = np.load('h2_222.npy')  # [k1, k2, k3, nmo, nmo, nmo, nmo]
    for kp, kq, kr in product(range(nkpts), repeat=3):
        ks = kconserv[kp, kq, kr]
        assert np.allclose(h2[kp, kq, kr], h2[ks, kr, kq].transpose(3, 2, 1, 0).conj())
        assert np.allclose(h2[kp, kq, kr], h2[kr, ks, kp].transpose(2, 3, 0, 1))
        assert np.allclose(h2[kp, kq, kr], h2[kr, ks, kp].transpose(2, 3, 0, 1))

    print("Constructing Super mat")
    total_kpts_considered = 0
    v2mat = np.zeros(((nkpts * nmo)**2, (nkpts * nmo)**2), dtype=np.complex128)
    v2tensor = np.zeros((nmo * nkpts, nmo * nkpts, nmo * nkpts, nmo * nkpts), dtype=np.complex128)
    for kp, kq, kr in product(range(nkpts), repeat=3):
        ks = kconserv[kp, kq, kr]
        for p, q, r, s in product(range(nmo), repeat=4):
            pkp = kp * nkpts + p
            qkq = kq * nkpts + q
            rkr = kr * nkpts + r
            sks = ks * nkpts + s
            v2tensor[pkp, qkq, rkr, sks] = h2[kp, kq, kr][p, q, r, s]

    assert np.allclose(v2tensor, v2tensor.transpose(2, 3, 0, 1))
    assert np.allclose(v2tensor, v2tensor.transpose(3, 2, 1, 0).conj())
    assert np.allclose(v2tensor, v2tensor.transpose(1, 0, 3, 2).conj())
    v2mat = v2tensor.transpose(0, 1, 3, 2).reshape(((nkpts * nmo)**2, (nkpts * nmo)**2))
    assert np.allclose(v2mat, v2mat.conj().T)

    w, v = np.linalg.eigh(v2mat)
    pos_w = np.where(w > 1.0E-8)[0][::-1]
    L = np.zeros((v2mat.shape[0], len(pos_w)), dtype=np.complex128)
    Ltensor = np.zeros((len(pos_w), nmo * nkpts, nmo * nkpts), dtype=np.complex128)
    for idx, widx in enumerate(pos_w):
        L[:, idx] = np.sqrt(w[widx]) * v[:, widx]
        Ltensor[idx, :, :] = np.sqrt(w[widx]) * v[:, widx].reshape((nmo * nkpts, nmo * nkpts))

    print(np.allclose(np.einsum('Lpq,Lsr->pqrs', Ltensor, Ltensor.conj()), v2tensor))
    print(np.allclose(L @ L.conj().T, v2mat))
    assert np.allclose(L @ L.conj().T, v2mat)





if __name__ == "__main__":
    main()