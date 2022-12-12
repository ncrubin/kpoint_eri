from itertools import product
import numpy as np
from pyscf.pbc import gto, cc, scf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf import ao2mo


import numpy as np

from pyscf import pbc# , scf, ao2mo

from pyscf.pbc.scf.chkfile import load_scf
from pyscf.pbc.tools import k2gamma

def checkpoint_to_meanfield(filename, n_kpts = [1, 1, 1]):
    """
    
    convert a pbc mean-field checkpoint file to a molecular mean-field object

    :param filename: path to the checkpoint file
    :param n_kpts: number of k-points (for determining the madelung constant)

    :return mf: the molecular mean-field object
    :return mol: the corresonding molecule
    :return madelung: the madelung constant
    
    """

    # load data from checkpoint file into a pbc mean-field object
    mol, data = load_scf(filename)
    kmf = pbc.scf.KRHF(mol)
    kmf.e_tot = data['e_tot']
    kmf.mo_coeff = data['mo_coeff']
    kmf.mo_energy = data['mo_energy']
    kmf.kpts = data['kpts']
    kmf.mo_occ = data['mo_occ']

    # calculate the madelung constant
    kpts = kmf.cell.make_kpts(n_kpts, wrap_around = True)
    madelung = pbc.tools.pbc.madelung(kmf.cell, kpts)

    # convert pbc mean-field object with k to pbc mean-field object at gamma point
    kmf_gamma = k2gamma.k2gamma(kmf)

    # run SCF with the (gamma-point) pbc mean-field object
    kmf_gamma.kernel()
    
    # run MP2 so we can be sure the molecular mean-field object gives consistent results
    kmf_gamma.MP2().run()

    # build a molecular mean-field object
    mol = kmf_gamma.cell
    mf = scf.RHF(mol)
    
    hcore = np.einsum('ui,vj,uv', kmf_gamma.mo_coeff, kmf_gamma.mo_coeff, kmf_gamma.get_hcore())
    
    nbf = len(kmf_gamma.mo_energy)

    mf.get_hcore = lambda *args: np.asarray(hcore)
    mf.get_ovlp = lambda *args: np.eye(nbf)
    mf.energy_nuc = lambda *args: kmf_gamma.energy_nuc()
    mf.mo_coeff = kmf_gamma.mo_coeff
    mf.mo_occ = kmf_gamma.mo_occ
    
    # now try to compute SCF energy by hand, in the MO basis
    
    eri_4fold_sym = kmf_gamma.with_df.ao2mo(kmf_gamma.mo_coeff)
    eri_8fold_sym = ao2mo.addons.restore('s8', eri_4fold_sym, nbf)
    eri = ao2mo.addons.restore('s1', eri_8fold_sym, nbf)

    print('symmetrize by hand...')
    ij = 0
    for i in range (0, nbf):
        for j in range (i, nbf):
            kl = 0
            for k in range (0, nbf):
                for l in range (k, nbf):
                    if ij <= kl :
                        tmp = eri[i, j, k, l] \
                            + eri[i, j, l, k] \
                            + eri[j, i, k, l] \
                            + eri[j, i, l, k] \
                            + eri[k, l, i, j] \
                            + eri[l, k, i, j] \
                            + eri[k, l, j, i] \
                            + eri[l, k, j, i]
                        tmp *= 0.125
                        eri[i, j, k, l] = tmp
                        eri[i, j, l, k] = tmp
                        eri[j, i, k, l] = tmp
                        eri[j, i, l, k] = tmp
                        eri[k, l, i, j] = tmp
                        eri[l, k, i, j] = tmp
                        eri[k, l, j, i] = tmp
                        eri[l, k, j, i] = tmp
                    kl += 1
            ij += 1
    print('done!')
    mf._eri = eri
    
    # compute SCF energy by hand
    num_alpha = int(np.sum(mf.mo_occ) / 2)
    num_beta = int(np.sum(mf.mo_occ) / 2)
    scf_energy = mf.energy_nuc() + \
                 2*np.einsum('ii',hcore[:num_alpha,:num_alpha]) + \
                 2*np.einsum('iijj',eri[:num_alpha,\
                                        :num_alpha,\
                                        :num_alpha,\
                                        :num_alpha]) - \
                   np.einsum('ijji',eri[:num_alpha,\
                                        :num_alpha,\
                                        :num_alpha,\
                                        :num_alpha])
    #print('scf energy by hand', scf_energy)

    # run molecular mean-field kernel
    mf.kernel()
    
    # shift orbital energies by the madelung constant
    for i in range (0, num_alpha):
        mf.mo_energy[i] -= madelung

    return mf, mol, madelung

def main():
    kmesh = [1, 1, 2]
    cell = gto.M(
        unit='B',
        a=[[0., 3.37013733, 3.37013733],
           [3.37013733, 0., 3.37013733],
           [3.37013733, 3.37013733, 0.]],
        mesh=[24,]*3,
        atom='''C 0 0 0
                  C 1.68506866 1.68506866 1.68506866''',
        basis='gth-szv',
        pseudo='gth-pade',
        verbose=4
    )
    cell.build()

    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts = kpts).rs_density_fit()
    mf.chkfile = 'ncr_c2.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

    norb = cell.nao

    sc_mf = k2gamma.k2gamma(mf) 
    scnorb = sc_mf.mo_coeff.shape[-1]
    kp, kq, kr, ks = [0, 0, 0, 0]

    orb = sc_mf.mo_coeff
    eri_4fold = ao2mo.kernel(sc_mf.mol, orb) # get the integrals in 4-fold symm format
    eri_8fold = ao2mo.restore('s8', eri_4fold, norb=scnorb) # get 8-fold symmetric integrals as a vector
    eri_8fold = ao2mo.restore(1, eri_8fold, norb=scnorb) # restore 8-fold symmetric integral vector to 4-tensor
    eri = ao2mo.kernel(sc_mf.mol, orb, aosym=1) # test if directly outputing 1 sym is sufficient
    assert np.allclose(eri_8fold, eri.reshape((scnorb,)*4)) # sans last two column permute but okay by symmetry

    # test 8-fold symmetry with 7 equalities
    assert np.allclose(eri_8fold, eri_8fold.transpose(0, 1, 3, 2))
    assert np.allclose(eri_8fold, eri_8fold.transpose(1, 0, 2, 3))
    assert np.allclose(eri_8fold, eri_8fold.transpose(1, 0, 3, 2))
    assert np.allclose(eri_8fold, eri_8fold.transpose(3, 2, 0, 1))
    assert np.allclose(eri_8fold, eri_8fold.transpose(2, 3, 0, 1))
    assert np.allclose(eri_8fold, eri_8fold.transpose(3, 2, 1, 0))
    assert np.allclose(eri_8fold, eri_8fold.transpose(2, 3, 1, 0))

    # form matrix representation of the integrals
    eri_mat = eri_8fold.transpose((0, 1, 3, 2)).reshape((scnorb**2, scnorb**2))
    assert np.allclose(eri_mat.T, eri_mat)

    # diagonalize with eigh
    w, u = np.linalg.eigh(eri_mat)

    # test if we are actually diagonalizing 
    assert np.allclose(u @ np.diag(w) @ u.T, eri_mat)

    sc_ed_mf, sc_ed_mol, madelung = checkpoint_to_meanfield('ncr_c2.chk')
    print(sc_ed_mf.energy_tot())
    print(mf.energy_tot())
    # u, s, vh = np.linalg.svd(eri_mat)
    # # get cholrank
    # widx = np.where(np.abs(w) > 1.0E-8)[0]
    # print(widx)
    # exit()

    # # populate cholesky tesnor and mat
    # chol = np.zeros((len(widx), scnorb, scnorb) , dtype=float)    
    # chol_mat = vh.T[:, widx]
    # for idx, ii in enumerate(widx):
    #     chol[idx, :, :] = np.sqrt(w[ii]) * u[:, ii].reshape((scnorb, scnorb))
    #     chol_mat[:, idx] *= np.sqrt(w[ii])
    #     assert np.allclose(chol[idx, :, :], chol[idx, :, :].T)

    # # reconstruct integrals from chol tensor (scnorb, scnorb, scnaux)
    # assert np.allclose(np.einsum('nij,nlk->ijkl', chol, chol), eri_8fold)



    # print(mf.energy_nuc())
    # print(sc_mf.energy_nuc() / np.prod(kmesh))
    # # 0.4697454477270757
    # # -1.87898179091
    # print(sc_mf.energy_tot() / np.prod(kmesh))
    # print(mf.energy_tot()) 

    # sc_mf.get_hcore()


if __name__ == "__main__":
    main()