"""

THIS IS A USING BTAS EXAMPLE


# calculate the madelung constant
kpts = kmf.cell.make_kpts(n_kpts, wrap_around = True)
madelung = pbc.tools.pbc.madelung(kmf.cell, kpts)

"""
import h5py
from itertools import product
import numpy as np
from pyscf.pbc import gto, cc, scf, mp
from pyscf import gto as gto_mol
from pyscf import scf as scf_mol
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf import ao2mo


import numpy as np

from pyscf import pbc# , scf, ao2mo

from pyscf.pbc.scf.chkfile import load_scf
from pyscf.pbc.tools import k2gamma

import time

try:
    import pybtas
except ImportError:
    raise ImportError(
        "pybtas could not be imported. Is it installed and in PYTHONPATH?")


def get_gamma_point(uc_mf, rerun_scf=True):
    if uc_mf.exxdiv is not None:
        raise ValueError("Currently this function will only work for exxdiv=None")

    sc_mf = k2gamma.k2gamma(uc_mf) 
    sc_cell = sc_mf.cell
    sc_kpts = sc_cell.make_kpts([1, 1, 1])
    sc_mf = scf.KRHF(sc_cell, kpts=sc_kpts, exxdiv=None).rs_density_fit()

    if rerun_scf: 
        sc_mf.kernel()
        assert np.isclose(sc_mf.energy_tot() / len(uc_mf.kpts), uc_mf.e_tot)
    else:
        raise ValueError("We really should rerun an SCF")

    scnorb = sc_mf.mo_coeff[0].shape[-1]
    eri = sc_mf.with_df.ao2mo([sc_mf.mo_coeff[0]]*4)
    eri = ao2mo.restore(1, eri, norb=scnorb)

    hcore_ao = sc_mf.get_hcore()[0]
    hcore_mo = sc_mf.mo_coeff[0].T @ hcore_ao @ sc_mf.mo_coeff[0]
    sc_ecore = sc_mf.energy_nuc()
    num_alpha = int(np.sum(sc_mf.mo_occ[0])) // 2
    num_beta = int(np.sum(sc_mf.mo_occ[0])) // 2
    n_orb = hcore_mo.shape[0]
    alpha_diag = [1] * num_alpha + [0] * (n_orb - num_alpha)
    beta_diag = [1] * num_beta + [0] * (n_orb - num_beta)
    scf_energy = sc_mf.energy_nuc() + \
             2*np.einsum('ii',hcore_mo[:num_alpha,:num_alpha]) + \
             2*np.einsum('iijj',eri[:num_alpha,\
                                    :num_alpha,\
                                    :num_alpha,\
                                    :num_alpha]) - \
               np.einsum('ijji',eri[:num_alpha,\
                                    :num_alpha,\
                                    :num_alpha,\
                                    :num_alpha])
    assert np.isclose(scf_energy, sc_mf.e_tot)

    # build molecular object
    mol = gto_mol.M()
    mol.nelectron = num_alpha + num_beta
    mf_mol = scf_mol.RHF(mol)
    mf_mol.get_hcore = lambda *args: np.asarray(hcore_mo)
    mf_mol.get_ovlp = lambda *args: np.eye(hcore_mo.shape[0])
    mf_mol.energy_nuc = lambda *args: sc_ecore
    mf_mol._eri = eri  # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)
    mf_mol.e_tot = scf_energy

    mf_mol.init_guess = '1e'
    mf_mol.mo_coeff = np.eye(hcore_mo.shape[0])
    mf_mol.mo_occ = np.array(alpha_diag) + np.array(beta_diag)
    mf_mol.mo_energy, _ = np.linalg.eigh(mf_mol.get_fock())

    return mf_mol

def get_cholesky_factors(eri_tensor: np.ndarray):
    """
    :param eri_tensor: 8-fold symmetry real valued chemist ordered ERI tensor
    :returns: Cholesky factor (naux, norb, norb)
    """
    eri = eri_tensor
    norb = eri.shape[0]
    # this checks if we have molecular level symmetry
    assert np.allclose(eri, eri.transpose(1, 0, 3, 2))
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1))
    assert np.allclose(eri, eri.transpose(1, 0, 2, 3))
    assert np.allclose(eri, eri.transpose(0, 1, 2, 3))

    # form matrix representation of the integrals
    eri_mat = eri.transpose((0, 1, 3, 2)).reshape((norb**2, norb**2))
    assert np.allclose(eri_mat.T, eri_mat)

    # diagonalize with eigh
    w, u = np.linalg.eigh(eri_mat)
    w = w[::-1]
    u = u[:, ::-1]
    w_nz= np.where(w > 1.0E-8)[0]

    # test if we are actually diagonalizing 
    assert np.allclose(u @ np.diag(w) @ u.T, eri_mat)

    # populate cholesky tesnor and mat
    chol = np.zeros((len(w_nz), norb, norb) , dtype=float)    
    chol_mat = np.zeros((len(w_nz), norb**2), dtype=float)
    for idx, ii in enumerate(w_nz):
        chol[idx, :, :] = np.sqrt(w[ii]) * u[:, ii].reshape((norb, norb))
        assert np.allclose(chol[idx, :, :], chol[idx, :, :].T)
        chol_mat[idx, :] = np.sqrt(w[ii]) * u[:, ii]

    # # reconstruct integrals from chol tensor (scnorb, scnorb, scnaux)
    assert np.allclose(np.einsum('nij,nlk->ijkl', chol, chol), eri)
    return chol, chol_mat


def main():
    kmesh = [1, 1, 3]
    cell = gto.M(
        unit='B',
        a=[[0., 3.37013733, 3.37013733],
           [3.37013733, 0., 3.37013733],
           [3.37013733, 3.37013733, 0.]],
        atom='''C 0 0 0
                  C 1.68506866 1.68506866 1.68506866''',
        basis='gth-szv',
        pseudo='gth-hf-rev',
        verbose=4
    )
    cell.build()
    
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts = kpts, exxdiv=None).rs_density_fit()
    mf.chkfile = 'ncr_c2.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mf_mol =  get_gamma_point(mf, rerun_scf=True)
    chol, chol_mat = get_cholesky_factors(mf_mol._eri)
    norb = chol.shape[1]
    chol = chol.transpose(1, 2, 0)
    chol = np.ascontiguousarray(chol) # (norb, norb, naux)
    chol_mat = np.ascontiguousarray(chol_mat.T)


    l2_norms = []
    abs_diff_eri = []
    c_vals = []
    cholesky_from_cp3 = []
    eris_from_cp3 = []
    eri_deltas = []
    mean_abs_deviation = []
    beta_cp3 = []
    gamma_cp3 = []
    scale_cp3 = []
    for c in range(5, 16):
        start_time = time.time()  # timing results if requested by user
        beta, gamma, scale = pybtas.cp3_from_cholesky(chol_mat.copy(),
                                                      c * norb,
                                                      random_start=False,
                                                      conv_eps=float(1.0E-5),
                                                      )
        cp3_calc_time = time.time() - start_time
        print("CP3 time ", cp3_calc_time)
        u_alpha_test = np.einsum("ar,br,xr,r->abx", beta, beta, gamma,
                                 scale.ravel())
        print("\tu_alpha l2-norm ", np.linalg.norm(u_alpha_test - chol))
        c_vals.append(c)
        cp3_eri = np.einsum('ijn,lkn->ijkl', u_alpha_test, u_alpha_test) 
        eris_from_cp3.append(cp3_eri)
        mean_abs_deviation.append(np.mean(np.abs(cp3_eri - mf_mol._eri)))
        eri_deltas.append(cp3_eri - mf_mol._eri)
        cholesky_from_cp3.append(u_alpha_test)
        beta_cp3.append(beta)
        gamma_cp3.append(gamma)
        scale_cp3.append(scale)
        l2_norms.append(np.linalg.norm(u_alpha_test - chol))
        print("mean_abs_deviation ", mean_abs_deviation[-1])
        print("cval ", c)
        print()


        with h5py.File('diamond_cp3_c{}.h5'.format(c), 'w') as fid:
            fid.create_dataset(name='chol_l2_norms', data=l2_norms)
            fid.create_dataset(name='c_vals', data=c)
            fid.create_dataset(name='eris_from_cp3', data=eris_from_cp3[-1])
            fid.create_dataset(name='mean_abs_deviation', data=mean_abs_deviation[-1])
            fid.create_dataset(name='eri_deltas', data=eri_deltas[-1])
            fid.create_dataset(name='cholesky_from_cp3', data=cholesky_from_cp3[-1])
            fid.create_dataset(name='beta_cp3', data=beta_cp3[-1])
            fid.create_dataset(name='gamma_cp3', data=gamma_cp3[-1])
            fid.create_dataset(name='scale_cp3', data=scale_cp3[-1])


    import matplotlib.pyplot as plt
    plt.semilogy(c_vals, mean_abs_deviation, 'C0o-')
    plt.xlabel("c", fontsize=14)
    plt.ylabel(r"$\langle |(pq|rs)_{cp3} - (pq|rs)|\rangle$", fontsize=14)
    plt.savefig("diamond_szv_113_mean_abs_deviation_hosvd.png", format='PNG', dpi=300
    )




if __name__ == "__main__":
    main()