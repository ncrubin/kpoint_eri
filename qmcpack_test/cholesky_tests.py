import numpy as np
import time
import scipy.linalg

from afqmctools.hamiltonian.converter import read_qmcpack_hamiltonian
from pauxy.utils.io import read_qmcpack_wfn_hdf

def one_rdm(A, B):
    O = A.conj().T @ B
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB.T

def get_kp_block(tensor, dims, kpoints):
    assert len(kpoints) == len(tensor.shape)
    slices = tuple(slice(ik*dim, (ik+1)*dim) for (ik, dim) in zip(kpoints, dims))
    return tensor[slices]

def test_kp_chol():
    hamil = read_qmcpack_hamiltonian('hamiltonian.h5')
    chol_vecs = hamil['chol']
    nmo_per_kp = hamil['nmo_pk']
    nchol_per_kp = hamil['nchol_pk']
    qk_to_k2 = hamil['qk_k2']
    nelec = hamil['nelec']
    nk = len(hamil['kpoints'])
    nelec_per_kp = nelec[0] // nk
    wfn, psi0 = read_qmcpack_wfn_hdf('hamiltonian.h5')
    print(nchol_per_kp)
    # for i in range(nk):
        # print(get_kp_block(psi0, [nmo_per_kp[i], nelec_per_kp], [i,i]))
    opdm = one_rdm(psi0[:,:nelec[0]], psi0[:,:nelec[0]])
    LQikn = []
    for i, c in enumerate(chol_vecs):
        LQikn.append(c.reshape((nk, nmo_per_kp[i], nmo_per_kp[i], nchol_per_kp[i])))
    hcore = hamil['hcore']
    enuc = hamil['enuc']
    e1b = 0.0
    hcore_sc = scipy.linalg.block_diag(*hcore)
    print(2*np.einsum('pq,pq->', hcore_sc, opdm)/8)
    # print(hcore_sc.shape)
    for ik in range(nk):
        e1b += np.einsum(
                'pq,pq->',
                hcore[ik],
                get_kp_block(opdm, [nmo_per_kp[ik]]*2, [ik]*2)
                )
    print(2*e1b/nk)
    ecoul = 0.0
    ecoul_2 = 0.0
    exx   = 0.0
    vj = [[0]*nk,[0]*nk]
    XL = [[0]*nchol_per_kp[i] for i in range(nk)]
    XR = [[0]*nchol_per_kp[i] for i in range(nk)]
    for Q in range(nk):
        LQ = LQikn[Q]
        for k1 in range(nk):
            for k3 in range(nk):
                k2 = qk_to_k2[Q][k1]
                k4 = qk_to_k2[Q][k3]
                TL = np.einsum(
                        'prn,ps->rsn',
                        LQ[k1],
                        get_kp_block(opdm, [nmo_per_kp[k1], nmo_per_kp[k3]], [k1, k3])
                        )
                TR = np.einsum(
                        'sqn,qr->srn',
                        LQ[k3].conj(),
                        get_kp_block(opdm, [nmo_per_kp[k2], nmo_per_kp[k4]], [k4, k2])
                        )
                exx += np.einsum('rsn,srn->', TL, TR)
        for k1 in range(nk):
            k2 = qk_to_k2[Q][k1]
            XL[Q] += np.einsum(
                    'pqn,pq->n',
                    LQ[k1],
                    get_kp_block(opdm, [nmo_per_kp[k1], nmo_per_kp[k2]], [k1, k2])
                    )
            XR[Q] += np.einsum(
                    'srn,rs->n',
                    LQ[k1].conj(),
                    get_kp_block(opdm, [nmo_per_kp[k2], nmo_per_kp[k1]], [k2, k1])
                    )
    for Q in range(nk):
        ecoul += np.dot(XL[Q], XR[Q])

    print((2*ecoul - exx)/nk)
    print(2*ecoul/nk, -exx/nk)

def test_kp_eri():
    hamil = read_qmcpack_hamiltonian('hamiltonian.h5')
    chol_vecs = hamil['chol']
    nmo_per_kp = hamil['nmo_pk']
    nchol_per_kp = hamil['nchol_pk']
    qk_to_k2 = hamil['qk_k2']
    nelec = hamil['nelec']
    nk = len(hamil['kpoints'])
    nelec_per_kp = nelec[0] // nk
    wfn, psi0 = read_qmcpack_wfn_hdf('hamiltonian.h5')
    opdm = one_rdm(psi0[:,:nelec[0]], psi0[:,:nelec[0]])
    LQikn = []
    for i, c in enumerate(chol_vecs):
        LQikn.append(c.reshape((nk, nmo_per_kp[i], nmo_per_kp[i], nchol_per_kp[i])))

    # Test ERIs.
    from pyscf.pbc.df import FFTDF
    from afqmctools.utils.pyscf_utils import load_from_pyscf_chk
    scf_data = load_from_pyscf_chk('diamond_222.chk')
    df = FFTDF(scf_data['cell'], kpts=scf_data['kpts'])
    kpoints = scf_data['kpts']
    mo_coeff = scf_data['mo_coeff']
    nmo = mo_coeff[0].shape[1] # safe for the moment
    num_kp = len(kpoints)
    # too slow in python just sample some randomly
    nsamp = 3
    kp_idx = np.arange(num_kp)
    qs = np.random.choice(kp_idx, nsamp, replace=False)
    k1s = np.random.choice(kp_idx, nsamp, replace=False)
    k4s = np.random.choice(kp_idx, nsamp, replace=False)
    for iq in qs:
        for k1 in k1s:
            k2 = qk_to_k2[iq][k1]
            for k4 in k4s:
                k3 = qk_to_k2[iq][k4]
                kpt1234 = [kpoints[ik] for ik in [k1,k2,k3,k4]]
                mos1234  = [mo_coeff[ik] for ik in [k1,k2,k3,k4]]
                eri_pqrs = df.ao2mo(mos1234, kpts=kpt1234, compact=False) / num_kp
                eri_chol = np.einsum(
                                'pqn,srn->pqrs',
                                LQikn[iq][k1],
                                LQikn[iq][k4].conj(),
                                optimize=True).reshape((nmo*nmo,)*2)
                # print("Q = {:d}, k1 = {:d} k4 = {:d}".format(iq, k1, k4), np.linalg.norm(eri_pqrs-eri_chol))
                assert np.linalg.norm(eri_pqrs-eri_chol) < 1e-12



if __name__ == '__main__':
    # test_kp_chol()
    test_kp_eri()
