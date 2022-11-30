from functools import reduce
import os
import numpy as np
from itertools import product

from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.df import FFTDF

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

from kpoint_eri.resource_estimates.df.compute_lambda_df import compute_lambda_ncr, compute_lambda_ncr_v2
from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABV2KpointIntegrals
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from pyscf.pbc import ao2mo, tools
from pyscf.pbc.lib.kpts_helper import get_kconserv



_file_path = os.path.dirname(os.path.abspath(__file__))

def test_compute_lambda_df():
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    mo_coeffs = kmf.mo_coeff
    num_kpoints = len(mo_coeffs)
    kpoints = ham['kpoints']
    momentum_map = ham['qk_k2']
    chol = ham['chol']
    df_factors = df.double_factorize(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=1e-5)
    lambda_tot, lambda_T, lambda_F = df.compute_lambda(
            ham['hcore'],
            df_factors,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    df_factors = df.double_factorize_batched(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=1e-5)
    lambda_tot_batch, lambda_T_, lambda_F_ = df.compute_lambda(
            ham['hcore'],
            df_factors,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    assert lambda_tot - lambda_tot_batch < 1e-12

def lambda_calc():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABKpointIntegrals
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0E-4)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda_ncr(hcore_mo, helper)
    print(lambda_tot)
    print(num_eigs)

def lambda_v2_calc():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    from pyscf.pbc.scf.chkfile import load_scf
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh)

    mf = scf.KRHF(cell, kpts).rs_density_fit()
    # mydf = FFTDF(mf.cell, kpts=kpts)
    # mf.with_df = mydf
    mf.chkfile = 'fft_ncr_test_C_density_fitints.chk'
    # cell, scf_dict = load_scf(mf.chkfile)
    # mf.e_tot = scf_dict['e_tot']
    # mf.kpts = scf_dict['kpts']
    # mf.mo_coeff = scf_dict['mo_coeff']
    # mf.mo_energy = scf_dict['mo_energy']
    # mf.mo_occ = scf_dict['mo_occ']
    # mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()


    mymp = mp.KMP2(mf)
    mymp.density_fit()
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABV2KpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0E-2)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda_ncr_v2(hcore_mo, helper)
    print(lambda_tot)
    print(num_eigs)

    # from pyscf.pbc.tools.k2gamma import k2gamma
    # supercell_mf = k2gamma(mf)
    # # supercell_mf.kernel()
    # supercell_mf.energy_elec()
    # supercell_mf.e_tot = supercell_mf.energy_tot()
    # print(supercell_mf.e_tot / np.prod(kmesh))
    # print(mf.e_tot)
    # print()
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    from pyscf.pbc.df import RSDF
    supercell_mf = k2gamma(mf, make_real=False)
    supercell_mf.verbose = 10
    mydf = RSDF(supercell_mf.cell, supercell_mf.kpts)
    # mydf.mesh = [7, 7, 7 * 3]
    # mydf.mesh_compact = [7, 7, 7 * 3]
    # mydf.omega = 0.522265625
    supercell_mf.with_df = mydf
    # supercell_mf.max_cycle = 1
    # supercell_mf.kernel()
    supercell_mf.energy_elec()
    supercell_mf.e_tot = supercell_mf.energy_tot()
    print(supercell_mf.e_tot / np.prod(kmesh))
    print(mf.e_tot)
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = DFABV2KpointIntegrals(cholesky_factor=supercell_Luv, kmf=supercell_mf)
    supercell_helper.double_factorize(thresh=1.0E-13)
    sc_nk = supercell_helper.nk
    sc_help = supercell_helper
 
    for kidx in range(sc_nk):
        for kpidx in range(sc_nk):
            for qidx in range(sc_nk):                 
                kmq_idx = supercell_helper.k_transfer_map[qidx, kidx]
                kpmq_idx = supercell_helper.k_transfer_map[qidx, kpidx]
                exact_eri_block = supercell_helper.get_eri_exact([kidx, kmq_idx, kpmq_idx, kpidx])
                test_eri_block = supercell_helper.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                # assert np.allclose(exact_eri_block, test_eri_block)
                print(np.allclose(exact_eri_block, test_eri_block))

    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body, sc_num_eigs = compute_lambda_ncr_v2(supercell_hcore_mo, sc_help)
    print(sc_lambda_one_body, lambda_one_body)
    print(sc_lambda_two_body, lambda_two_body)
    print(sc_num_eigs, num_eigs)
    exit()

    assert np.isclose(sc_lambda_one_body, lambda_one_body)
    assert np.isclose(sc_lambda_two_body, lambda_two_body)

def get_eri(mf, kpts=None,
        compact=True):         
    from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
    import numpy
    from pyscf import lib
    from pyscf.pbc import tools
    mydf = mf.with_df
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'fft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    q = kptj - kpti
    coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

    #:ao_pairs_G = get_ao_pairs_G(mydf, kptijkl[:2], q, compact=False)
    #:# ao_pairs_invG = rho_kl(-(G+k_ij)) = conj(rho_lk(G+k_ij)).swap(r,s)
    #:#=get_ao_pairs_G(mydf, [kptl,kptk], q, compact=False).transpose(0,2,1).conj()
    #:ao_pairs_invG = get_ao_pairs_G(mydf, -kptijkl[2:], q, compact=False).conj()
    #:ao_pairs_G *= coulG.reshape(-1,1)
    #:eri = lib.dot(ao_pairs_G.T, ao_pairs_invG, cell.vol/ngrids**2)
    if is_zero(kpti-kptl) and is_zero(kptj-kptk):
        if is_zero(kpti-kptj):
            aoi = mydf._numint.eval_ao(cell, coords, kpti)[0]
            aoi = aoj = numpy.asarray(aoi.T, order='C')
        else:
            aoi, aoj = mydf._numint.eval_ao(cell, coords, kptijkl[:2])
            aoi = numpy.asarray(aoi.T, order='C')
            aoj = numpy.asarray(aoj.T, order='C')
        aos = (aoi, aoj, aoj, aoi)
    else:
        aos = mydf._numint.eval_ao(cell, coords, kptijkl)
        aos = [numpy.asarray(x.T, order='C') for x in aos]
    fac = numpy.exp(-1j * numpy.dot(coords, q))
    max_memory = max_memory - aos[0].nbytes*4*1e-6
    eri = _contract_plain(mydf, aos, coulG, fac, max_memory=max_memory)
    return eri

def _contract_plain(mydf, mos, coulG, phase, max_memory):
    import numpy 
    from pyscf import lib
    from pyscf.pbc import tools
    cell = mydf.cell
    moiT, mojT, mokT, molT = mos
    nmoi, nmoj, nmok, nmol = [x.shape[0] for x in mos]
    ngrids = moiT.shape[1]
    wcoulG = coulG * (cell.vol/ngrids)
    dtype = numpy.result_type(phase, *mos)
    eri = numpy.empty((nmoi*nmoj,nmok*nmol), dtype=dtype)

    blksize = int(min(max(nmoi,nmok), (max_memory*1e6/16 - eri.size)/2/ngrids/max(nmoj,nmol)+1))
    assert blksize > 0
    buf0 = numpy.empty((blksize,max(nmoj,nmol),ngrids), dtype=dtype)
    buf1 = numpy.ndarray((blksize,nmoj,ngrids), dtype=dtype, buffer=buf0)
    buf2 = numpy.ndarray((blksize,nmol,ngrids), dtype=dtype, buffer=buf0)
    for p0, p1 in lib.prange(0, nmoi, blksize):
        mo_pairs = numpy.einsum('ig,jg->ijg', moiT[p0:p1].conj()*phase,
                                mojT, out=buf1[:p1-p0])
        mo_pairs_G = tools.fft(mo_pairs.reshape(-1,ngrids), mydf.mesh)
        mo_pairs = None
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        mo_pairs_G = None
        v *= phase.conj()
        if dtype == numpy.double:
            v = numpy.asarray(v.real, order='C')
        for q0, q1 in lib.prange(0, nmok, blksize):
            mo_pairs = numpy.einsum('ig,jg->ijg', mokT[q0:q1].conj(),
                                    molT, out=buf2[:q1-q0])
            eri[p0*nmoj:p1*nmoj,q0*nmol:q1*nmol] = lib.dot(v, mo_pairs.reshape(-1,ngrids).T)
        v = None
    return eri

def _format_kpts(kpts):
    import numpy
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    else:
        kpts = numpy.asarray(kpts)
        if kpts.size == 3:
            kptijkl = numpy.vstack([kpts]*4).reshape(4,3)
        else:
            kptijkl = kpts.reshape(4,3)
    return kptijkl

def _iskconserv(cell, kpts):
    import numpy
    dk = kpts[1] - kpts[0] + kpts[3] - kpts[2]
    if abs(dk).sum() < 1e-9:
        return True
    else:
        s = 1./(2*numpy.pi)*numpy.dot(dk, cell.lattice_vectors().T)
        s_int = s.round(0)
        return abs(s - s_int).sum() < 1e-9


def fftdf_reconstruct():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.mesh = [6, 6, 6]
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    mf = scf.KRHF(cell, kpts)#.rs_density_fit()
    mydf = FFTDF(mf.cell, kpts=kpts)
    mf.with_df = mydf
    mf.kernel()
    nmo = mf.mo_coeff[0].shape[-1]
    naux = mf.with_df.get_naoaux() // 2

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    from pyscf.pbc import tools
    import numpy 
    nao = cell.nao_nr()
    mydf = FFTDF(cell, kpts=kpts)
    Lpq_kpts = []
    for i, kpti in enumerate(kpts):
        Lpq_kpts.append([])
        for j, kptj in enumerate(kpts):
            q = kptj - kpti
            coulG = tools.get_coulG(cell, q)
            ngrids = len(coulG)
            # ao_pairs_G = mydf.get_mo_pairs_G([kpti,kptj], q, compact=False)
            ao_pairs_G = mf.with_df.get_mo_pairs_G([mf.mo_coeff[i], mf.mo_coeff[j]], kpts=mf.kpts[[i, j], :])
            ao_pairs_G *= numpy.sqrt(coulG*cell.vol/ngrids**2).reshape(-1,1)
            Lpq_kpts[i].append(ao_pairs_G.reshape(-1,nao,nao))
        
    Lpq_kpts = np.array(Lpq_kpts)
    print(Lpq_kpts.shape)
    exit()
    
    # kconserv = get_kconserv(cell, kpts)
    # Lrs_kpts = []
    # for i, kpti in enumerate(kpts):
    #     Lrs_kpts.append([])
    #     for j, kptj in enumerate(kpts):
    #         Lrs_kpts[i].append([])
    #         q = kptj - kpti
    #         coulG = tools.get_coulG(cell, q)
    #         ngrids = len(coulG)
    #         for k, kptk in enumerate(kpts):
    #             # Handle the wrap-around k-points
    #             l = kconserv[i,j,k]
    #             kptl = kpts[l]
    #             # ao_pairs_invG = mydf.get_ao_pairs_G([-kptk,-kptl], q, compact=False).conj()
    #             ao_pairs_invG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kidx], mf.mo_coeff[qidx]], kpts=mf.kpts[[kidx, qidx], :])

    #             ao_pairs_invG *= numpy.sqrt(coulG*cell.vol/ngrids**2).reshape(-1,1)
    #             Lrs_kpts[i][j].append(ao_pairs_invG.reshape(-1,nao,nao))

    Luv = np.zeros((nkpts, nkpts, naux, nmo, nmo), dtype=np.complex128)
    for kidx, qidx in product(range(len(kpts)), repeat=2):
        ijG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kidx], mf.mo_coeff[qidx]], kpts=mf.kpts[[kidx, qidx], :])
        q = kpts[kidx] - kpts[qidx]
        coulG = tools.get_coulG(cell, k=q, mesh=mf.with_df.mesh)
        weighted_coulG = coulG * cell.vol / naux**2
        ijG = np.einsum('ni,n->ni', ijG, np.sqrt(weighted_coulG))
        Luv[kidx, qidx, :, :, :] = ijG.reshape((-1, nmo, nmo))


    kconserv = get_kconserv(cell, kpts) 
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):                 
                ks = kconserv[kp, kq, kr]

                eri_kpt = mf.with_df.ao2mo([mf.mo_coeff[i] for i in (kp,kq,kr,ks)],
                                            [kpts[i] for i in (kp,kq,kr,ks)])
                eri_kpt = eri_kpt.reshape([nmo]*4)
                # kpti, kptj, kptk, kptl = kpts[[kp, kq, kr, ks], :]
                # q = kptj - kpti
                # coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
                # rsG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kr], mf.mo_coeff[ks]], kpts=mf.kpts[[kr, ks], :])
                # pqG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kp], mf.mo_coeff[kq]], kpts=mf.kpts[[kp, kq], :]).conj()
                # rsG *= coulG
                # print(rsG.shape)
                # exit()
                eri_test = np.einsum('npq,nsr->pqrs', Luv[kp, kq], Luv[ks, kr].conj(), optimize=True)
                print(np.linalg.norm(eri_test - eri_kpt))
                # assert np.allclose(eri_test, eri_kpt)



    




if __name__ == "__main__":
    # fftdf_reconstruct()
    lambda_v2_calc()
