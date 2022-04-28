import numpy as np
import time
import scipy.linalg
from pyscf.pbc.df import FFTDF
from afqmctools.utils.pyscf_utils import load_from_pyscf_chk
from afqmctools.hamiltonian.converter import read_qmcpack_hamiltonian

def get_kp_block(tensor, dims, kpoints):
    assert len(kpoints) == len(tensor.shape)
    slices = tuple(slice(ik*dim, (ik+1)*dim) for (ik, dim) in zip(kpoints, dims))
    return tensor[slices]

def test_kp_eri():
    # Test ERIs.
    scf_data = load_from_pyscf_chk('diamond_222.chk')
    df = FFTDF(scf_data['cell'], kpts=scf_data['kpts'])
    kpoints = scf_data['kpts']
    mo_coeff = scf_data['mo_coeff']
    nmo = mo_coeff[0].shape[1] # safe for the moment
    num_kp = len(kpoints)
    hamil = read_qmcpack_hamiltonian('hamiltonian.h5')
    qk_to_k2 = hamil['qk_k2']
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
                # (p k q k - Q | r k' - Q s k')
                k3 = qk_to_k2[iq][k4]
                kpt1234 = [kpoints[ik] for ik in [k1,k2,k3,k4]]
                mos1234  = [mo_coeff[ik] for ik in [k1,k2,k3,k4]]
                eri_pqrs = df.ao2mo(mos1234, kpts=kpt1234, compact=False).reshape((nmo,)*4)
                # (r k' - Q s k' | p k q k - Q)
                kpt1234 = [kpoints[ik] for ik in [k3,k4,k1,k2]]
                mos1234  = [mo_coeff[ik] for ik in [k3,k4,k1,k2]]
                eri_pqrs_trans = df.ao2mo(mos1234, kpts=kpt1234, compact=False).reshape((nmo,)*4)
                eri_pqrs_trans = eri_pqrs_trans.transpose((2,3,0,1))
                assert np.linalg.norm(eri_pqrs-eri_pqrs_trans) < 1e-12
                # (s k' r k' -Q | q k - Q p k)*
                kpt1234 = [kpoints[ik] for ik in [k4,k3,k2,k1]]
                mos1234  = [mo_coeff[ik] for ik in [k4,k3,k2,k1]]
                eri_pqrs_trans = df.ao2mo(mos1234, kpts=kpt1234, compact=False).reshape((nmo,)*4)
                eri_pqrs_trans = eri_pqrs_trans.transpose((3,2,1,0)).conj()
                assert np.linalg.norm(eri_pqrs-eri_pqrs_trans) < 1e-12
                # (q k - Q p k | s k' p k'-Q)*
                kpt1234 = [kpoints[ik] for ik in [k2,k1,k4,k3]]
                mos1234  = [mo_coeff[ik] for ik in [k2,k1,k4,k3]]
                eri_pqrs_trans = df.ao2mo(mos1234, kpts=kpt1234, compact=False).reshape((nmo,)*4)
                eri_pqrs_trans = eri_pqrs_trans.transpose((1,0,3,2)).conj()
                assert np.linalg.norm(eri_pqrs-eri_pqrs_trans) < 1e-12



if __name__ == '__main__':
    # test_kp_chol()
    test_kp_eri()
