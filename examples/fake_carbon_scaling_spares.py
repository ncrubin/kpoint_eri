import numpy as np
from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse
import time

import matplotlib.pyplot as plt
import scipy
import scipy.optimize
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']


def linear(x, a, c):
    return a * x + c


def fit_linear_log_log(x, y, last_n_points=None):
    if last_n_points is None:
        last_n_points = len(x)
    if last_n_points > len(x):
        return None
    log_x = np.log(x[-last_n_points:])
    log_y = np.log(y[-last_n_points:])
    try:
        popt, pcov = scipy.optimize.curve_fit(linear, log_x, log_y)
        return popt
    except np.linalg.LinAlgError:
        return None

def get_ones_eri_tensor(N, Nk):
    return np.ones((Nk, Nk, Nk, N, N, N, N))


def idealized_overcounting():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        [2, 2, 3],
        [2, 3, 3],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]
    nso = 6

    tof_per_step = []
    nk_vals = []
    n_vals = []
    for kmesh in kmesh_set:
        print(np.prod(kmesh))
        start_time = time.time()
        get_ones_eri_tensor(nso, np.prod(kmesh))
        end_time = time.time()
        print("time to generate kmesh {} eris ".format(kmesh), 
              end_time - start_time)
        nk = np.prod(kmesh)

        nnz = (nk**3) * nso**4
        res = cost_sparse(nso, nk, 1, nnz, 1.0e-3, 10, 20_000)
        print(res[0])
        tof_per_step.append(res[0])
        nk_vals.append(nk)
        n_vals.append(nso)

    tof_per_step = np.array(tof_per_step)
    nk_vals = np.array(nk_vals)
    n_vals = np.array(n_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    y_vals = tof_per_step / n_vals**2
    x_vals = nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    x = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    y = np.exp(b) * x**m
    ax.loglog(x, y, 'k-', alpha=1., label=r'uc: $\mathcal{{o}}(n_{{k}}^{{{:1.3f}}})$'.format(m))

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$n_{k}$", fontsize=14)
    ax.set_ylabel(r"toffolis / select + prepare + prepare$^{-1}$ / $n^{2}$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=2, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("fake_sparse_scaling.png", format='png', dpi=300)


def idealized_n_overcounting():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        [2, 2, 3],
        [2, 3, 3],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]
    nso = 6

    tof_per_step = []
    nk_vals = []
    n_vals = []
    nnz_nk = []
    nnz_vals = []
    sc_nnz_nk = []
    sc_nnz_vals = []
    for kmesh in kmesh_set:
        print(np.prod(kmesh))
        start_time = time.time()
        get_ones_eri_tensor(nso, np.prod(kmesh))
        end_time = time.time()
        print("time to generate kmesh {} eris ".format(kmesh), 
              end_time - start_time)
        nk = np.prod(kmesh)

        nnz = (nk**3) * nso**4
        res = cost_sparse(nso, nk, 1, nnz, 1.0e-3, 10, 20_000)
        print(res[0])
        tof_per_step.append(res[0])
        nk_vals.append(nk)
        n_vals.append(nso)
        nnz_nk.append(nnz / (nk**3))
        nnz_vals.append(nnz)

        sc_nnz_nk.append((nk * nso)**4 / nk**4 )
        sc_nnz_vals.append((nk * nso)**4)

    tof_per_step = np.array(tof_per_step)
    nk_vals = np.array(nk_vals)
    n_vals = np.array(n_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)


    sc_y_vals = sc_nnz_vals  #’/ np.array(n_vals)**4
    sc_x_vals = nk_vals
    sc_m, sc_b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    x = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    y = np.exp(sc_b) * x**sc_m
    ax.loglog(x, y, 'k-', alpha=1., label=r'SC: $\mathcal{{O}}(NNZ^{{{:1.3f}}})$'.format(sc_m))

    y_vals = nnz_vals # / np.array(n_vals)**4
    x_vals = nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    x = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    y = np.exp(b) * x**m
    ax.loglog(x, y, 'k-', alpha=1., label=r'UC: $\mathcal{{O}}(NNZ^{{{:1.3f}}})$'.format(m))

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"NNZ / $N^{4}$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("fake_sparse_NNZ_scaling.png", format='png', dpi=300)


def count_like_code(kmf, nso, return_nk_counter=False):
    from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk
    from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import unique_iter, unique_iter_pq_rs, unique_iter_pr_qs, unique_iter_ps_qr
    kpts_helper = KptsHelper(kmf.cell, kmf.kpts)
    nkpts = len(kmf.kpts)
    completed = np.zeros((nkpts,nkpts,nkpts), dtype=bool)
    counter = 0
    nk_counter = 0
    for kvals in loop_kkk(nkpts):
        kp, kq, kr = kvals
        ks = kpts_helper.kconserv[kp, kq, kr]
        if not completed[kp,kq,kr]:
            nk_counter += 1
            eri_block = np.ones((nso, nso, nso, nso)) 
            if kp == kq == kr == ks:
                completed[kp,kq,kr] = True
                # counter += nso**4
                counter += len(list(unique_iter(nso)))
                # print(np.count_nonzero(eri_block), nso**4, len(list(unique_iter(nso))))
                assert nso**4 > len(list(unique_iter(nso)))
                # for ftuple in unique_iter(nso):
                #     p, q, r, s = ftuple
                #     counter += np.count_nonzero(eri_block[p, q, r, s])
            elif kp == kq and kr == ks:
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                # counter += nso**4
                counter += len(list(unique_iter_pq_rs(nso)))
                # print(np.count_nonzero(eri_block), nso**4, len(list(unique_iter_pq_rs(nso))))
                assert nso**4 > len(list(unique_iter_pq_rs(nso)))


                # for ftuple in unique_iter_pq_rs(nso):
                #     p, q, r, s = ftuple
                #     counter += np.count_nonzero(eri_block[p, q, r, s])
            elif kp == ks and kq == kr:
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                # counter += nso**4
                counter += len(list(unique_iter_ps_qr(nso)))
                # print(np.count_nonzero(eri_block), nso**4, len(list(unique_iter_ps_qr(nso))))
                assert nso**4 > len(list(unique_iter_ps_qr(nso)))

                # for ftuple in unique_iter_ps_qr(nso):
                #     p, q, r, s = ftuple
                #     counter += np.count_nonzero(eri_block[p, q, r, s])
            elif kp == kr and kq == ks:
                completed[kp,kq,kr] = True
                completed[kq,kp,ks] = True
                # counter += nso**4
                counter += len(list(unique_iter_pr_qs(nso)))
                # print(np.count_nonzero(eri_block), nso**4, len(list(unique_iter_pr_qs(nso))))
                assert nso**4 > len(list(unique_iter_pr_qs(nso)))
                # for ftuple in unique_iter_pr_qs(nso):
                #     p, q, r, s = ftuple
                #     counter += np.count_nonzero(eri_block[p, q, r, s])
            else:
                counter += nso**4
                # print(np.count_nonzero(eri_block), nso**4)

                # counter += np.count_nonzero(eri_block)
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                completed[kq,kp,ks] = True
                completed[ks,kr,kq] = True
    assert completed.all()
    if return_nk_counter:
        return counter, nk_counter
    return counter

def true_nk_counting_routine():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        # [2, 2, 3],
        # [2, 3, 3],
        # [3, 3, 3],
        # [4, 4, 4],
        # [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]
    nso = 6

    from pyscf.pbc import gto, scf, mp, cc
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
    cell.verbose = 0
    cell.build()


    tof_per_step = []
    sc_tof_per_step = []
    Nk_vals = []
    nNk_vals = []
    N_vals = []
    sc_nNk_vals = []
    from tqdm import tqdm
    for kmesh in tqdm(kmesh_set):
        kpts = cell.make_kpts(kmesh)
        mf = scf.KRHF(cell, kpts).rs_density_fit()
        Nk = np.prod(kmesh)
        nnz, nnk = count_like_code(mf, nso, return_nk_counter=True)

        res = cost_sparse(nso, Nk, 1, nnz, 1.0E-3, 10, 20_000)
        tof_per_step.append(res[0])
        Nk_vals.append(Nk)
        N_vals.append(nso)
        nNk_vals.append(nnk)

        from pyscf.pbc import tools
        sc_cell = tools.super_cell(cell, kmesh)
        sc_mf = scf.KRHF(sc_cell, [[0, 0, 0]]).rs_density_fit()
        sc_nnz, sc_nnk = count_like_code(sc_mf, nso * Nk, return_nk_counter=True)
        sc_nNk_vals.append(sc_nnk)


        sc_res = cost_sparse(nso * Nk, 1, 1, sc_nnz, 1.0E-3, 10, 20_000)
        sc_tof_per_step.append(sc_res[0])

    tof_per_step = np.array(tof_per_step)
    sc_tof_per_step = np.array(sc_tof_per_step)
    Nk_vals = np.array(Nk_vals)
    N_vals = np.array(N_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sc_y_vals = sc_nNk_vals
    sc_x_vals = Nk_vals
    m, b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[1], mec=colors[1],  linestyle='',
              color=colors[1], alpha=0.5)
    X = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k--', alpha=1., label=r'SC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    y_vals = nNk_vals
    x_vals = Nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    X = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k-', alpha=1., label=r'UC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))


    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"Scaling of Nk with 4-fold symmetry", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("less_fake_sparse_Nk_scaling.png", format='PNG', dpi=300)

def true_N_counting_routine():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        # [2, 2, 3],
        # [2, 3, 3],
        # [3, 3, 3],
        # [4, 4, 4],
        # [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]
    nso = 6

    from pyscf.pbc import gto, scf, mp, cc
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
    cell.verbose = 0
    cell.build()


    tof_per_step = []
    sc_tof_per_step = []
    Nk_vals = []
    nNk_vals = []
    nnz_per_nk = []
    nnz_vals = []
    N_vals = []
    sc_nNk_vals = []
    sc_nnz_per_nk = []
    sc_nnz_vals = []
    from tqdm import tqdm

    for kmesh in tqdm(kmesh_set):
        kpts = cell.make_kpts(kmesh)
        mf = scf.KRHF(cell, kpts).rs_density_fit()
        Nk = np.prod(kmesh)
        nnz, nnk = count_like_code(mf, nso, return_nk_counter=True)

        res = cost_sparse(nso, Nk, 1, nnz, 1.0E-3, 10, 20_000)
        tof_per_step.append(res[0])
        Nk_vals.append(Nk)
        N_vals.append(nso)
        nNk_vals.append(nnk)
        nnz_per_nk.append(nnz/ nnk)
        nnz_vals.append(nnz)
 

        from pyscf.pbc import tools
        sc_cell = tools.super_cell(cell, kmesh)
        sc_mf = scf.KRHF(sc_cell, [[0, 0, 0]]).rs_density_fit()
        sc_nnz, sc_nnk = count_like_code(sc_mf, nso * Nk, return_nk_counter=True)
        sc_nNk_vals.append(sc_nnk)
        sc_nnz_per_nk.append(sc_nnz/ sc_nnk)

        sc_res = cost_sparse(nso * Nk, 1, 1, sc_nnz, 1.0E-3, 10, 20_000)
        sc_tof_per_step.append(sc_res[0])
        sc_nnz_vals.append(sc_nnz)

    tof_per_step = np.array(tof_per_step)
    sc_tof_per_step = np.array(sc_tof_per_step)
    Nk_vals = np.array(Nk_vals)
    N_vals = np.array(N_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sc_y_vals = np.array(sc_nnz_vals)#  / nso**4
    sc_x_vals = Nk_vals
    m, b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[1], mec=colors[1],  linestyle='',
              color=colors[1], alpha=0.5)
    X = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k--', alpha=1., label=r'SC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    y_vals = np.array(nnz_vals)#  / nso**4
    x_vals = Nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    X = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k-', alpha=1., label=r'UC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))


    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"NNZ", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("less_fake_sparse_N_scaling.png", format='PNG', dpi=300)
    print("DONE")



def true_counting_routine():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        # [2, 2, 3],
        # [2, 3, 3],
        # [3, 3, 3],
        # [4, 4, 4],
        # [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]
    nso = 6

    from pyscf.pbc import gto, scf, mp, cc
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
    cell.verbose = 0
    cell.build()


    tof_per_step = []
    sc_tof_per_step = []
    Nk_vals = []
    N_vals = []
    from tqdm import tqdm
    for kmesh in tqdm(kmesh_set):
        kpts = cell.make_kpts(kmesh)
        mf = scf.KRHF(cell, kpts).rs_density_fit()
        Nk = np.prod(kmesh)
        nnz = count_like_code(mf, nso)

        res = cost_sparse(nso, Nk, 1, nnz, 1.0E-3, 10, 20_000)
        tof_per_step.append(res[0])
        Nk_vals.append(Nk)
        N_vals.append(nso)

        from pyscf.pbc import tools
        sc_cell = tools.super_cell(cell, kmesh)
        sc_mf = scf.KRHF(sc_cell, [[0, 0, 0]]).rs_density_fit()
        sc_nnz = count_like_code(sc_mf, nso * Nk)

        sc_res = cost_sparse(nso * Nk, 1, 1, sc_nnz, 1.0E-3, 10, 20_000)
        sc_tof_per_step.append(sc_res[0])

    tof_per_step = np.array(tof_per_step)
    sc_tof_per_step = np.array(sc_tof_per_step)
    Nk_vals = np.array(Nk_vals)
    N_vals = np.array(N_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sc_y_vals = sc_tof_per_step / N_vals**2
    sc_x_vals = Nk_vals
    m, b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[1], mec=colors[1],  linestyle='',
              color=colors[1], alpha=0.5)
    X = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k--', alpha=1., label=r'SC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    y_vals = tof_per_step / N_vals**2
    x_vals = Nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    X = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k-', alpha=1., label=r'UC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))


    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"Toffolis / SELECT + PREPARE + PREPARE$^{-1}$ / $N^{2}$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("less_fake_sparse_scaling.png", format='PNG', dpi=300)

def carbon_counting_routine():
    kmesh_set = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 1],
        [2, 2, 3],
        [2, 3, 3],
        [3, 3, 3],
        # [4, 4, 4],
        # [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]

    from pyscf.pbc import gto, scf, mp, cc
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
    cell.verbose = 0
    cell.build()


    tof_per_step = []
    sc_tof_per_step = []
    Nk_vals = []
    N_vals = []
    sc_Nk_vals = []
    sc_N_vals = []

    from tqdm import tqdm
    from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

    for kmesh in tqdm(kmesh_set):
        kpts = cell.make_kpts(kmesh, scaled_center=[1/8, 1/8, 1/8])

        mf = scf.KRHF(cell, kpts).rs_density_fit()
        mf.kernel()

        mymp = mp.KMP2(mf)
        Luv = cholesky_from_df_ints(mymp)
        helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf, threshold=1.0E-5)
        Nk = np.prod(kmesh)
        nso = Luv[0, 0].shape[-1]
        nnz = helper.get_total_unique_terms_above_thresh()

        res = cost_sparse(nso, Nk, 1, nnz, 1.0E-3, 10, 20_000)
        tof_per_step.append(res[0])
        Nk_vals.append(Nk)
        N_vals.append(nso)
        
        # if Nk > 9:
        #     continue
        from pyscf.pbc import tools
        sc_cell = tools.super_cell(cell, kmesh)
        sc_kpts = sc_cell.make_kpts([1, 1, 1], scaled_center=[1/8, 1/8, 1/8])
        sc_mf = scf.KRHF(sc_cell, sc_kpts).rs_density_fit()
        sc_mf.kernel()
        sc_mymp = mp.KMP2(sc_mf)
        sc_Luv = cholesky_from_df_ints(sc_mymp)
        sc_nso = sc_Luv[0, 0].shape[-1]
        sc_helper = NCRSSparseFactorizationHelper(cholesky_factor=sc_Luv, kmf=sc_mf, threshold=1.0E-5)

        sc_nnz = sc_helper.get_total_unique_terms_above_thresh()
        sc_res = cost_sparse(sc_nso, 1, 1, sc_nnz, 1.0E-3, 10, 20_000)
        sc_tof_per_step.append(sc_res[0])
        sc_Nk_vals.append(Nk)
        sc_N_vals.append(nso)
 

    tof_per_step = np.array(tof_per_step)
    sc_tof_per_step = np.array(sc_tof_per_step)
    Nk_vals = np.array(Nk_vals)
    N_vals = np.array(N_vals)
    sc_Nk_vals = np.array(sc_Nk_vals)
    sc_N_vals = np.array(sc_N_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sc_y_vals = sc_tof_per_step / sc_N_vals**2
    sc_x_vals = sc_Nk_vals
    m, b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[1], mec=colors[1],  linestyle='',
              color=colors[1], alpha=0.5)
    X = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k--', alpha=1., label=r'SC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    y_vals = tof_per_step / N_vals**2
    x_vals = Nk_vals
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    X = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, 'k-', alpha=1., label=r'UC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))


    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"Toffolis / SELECT + PREPARE + PREPARE$^{-1}$ / $N^{2}$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("carbon_sparse_scaling.png", format='PNG', dpi=300)


def carbon_counting_nnz_per_nk_routine():
    kmesh_set = [
      [1, 1, 1],
      [1, 1, 2],
      [1, 2, 2],
      [1, 2, 3],
      [2, 2, 2],
      [3, 3, 1],
      [2, 2, 3],
      [2, 3, 3],
      [3, 3, 3],
      [4, 4, 4],
        # [5, 5, 5],
        # [6, 6, 6],
        # [7, 7, 7]
    ]

    from pyscf.pbc import gto, scf, mp, cc
    from pyscf.pbc.tools import pyscf_ase
    from ase.lattice.cubic import Diamond
    from ase.build import bulk

    # pp_name = "gth-hf-rev"
    # ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
    # cell = gto.Cell()
    # cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    # cell.a = ase_atom.cell[:].copy()
    # cell.basis = 'gth-szv'
    # cell.pseudo = "gth-hf-rev"
    # cell.verbose = 0
    # cell.build()
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
    cell.verbose = 0
    cell.build()


    tof_per_step = []
    sc_tof_per_step = []
    Nk_vals = []
    N_vals = []
    nnz_per_nk = []
    nnz_vals = []
    nnk_vals = []
    sc_Nk_vals = []
    sc_N_vals = []
    sc_nnz_per_nk = []
    sc_nnz_vals = []

    from tqdm import tqdm
    from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

    for kmesh in tqdm(kmesh_set):
        kpts = cell.make_kpts(kmesh, scaled_center=[1/8, 1/8, 1/8])
        mf = scf.KRHF(cell, kpts).rs_density_fit()
        mf.verbose = 4
        start_time = time.time()
        mf.kernel()
        end_time = time.time()
        print("SCF TIme K-point :", end_time-start_time)

        mymp = mp.KMP2(mf)
        Luv = cholesky_from_df_ints(mymp)
        helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf, threshold=1.0E-5)
        Nk = np.prod(kmesh)
        nso = Luv[0, 0].shape[-1]
        start_time = time.time()
        nnz, nnk = helper.get_total_unique_terms_above_thresh(return_nk_counter=True)
        end_time = time.time()
        print("nnz calculation k-point ", end_time - start_time)

        res = cost_sparse(nso, Nk, 1, nnz, 1.0E-3, 10, 20_000)
        tof_per_step.append(res[0])
        Nk_vals.append(Nk)
        N_vals.append(nso)
        nnz_per_nk.append(nnz/ nnk)
        nnz_vals.append(nnz)
        nnk_vals.append(nnk)
        
        # if Nk > 9:
        #     continue
        from pyscf.pbc import tools
        sc_cell = tools.super_cell(cell, kmesh)
        sc_kpts = sc_cell.make_kpts([1, 1, 1], scaled_center=[1/8, 1/8, 1/8])
        sc_mf = scf.KRHF(sc_cell, sc_kpts).rs_density_fit()
        sc_mf.verbose = 4
        start_time = time.time()
        sc_mf.kernel()
        end_time = time.time()
        print("SCF TIme SCt :", end_time-start_time)

        sc_mymp = mp.KMP2(sc_mf)
        sc_Luv = cholesky_from_df_ints(sc_mymp)
        sc_nso = sc_Luv[0, 0].shape[-1]
        sc_helper = NCRSSparseFactorizationHelper(cholesky_factor=sc_Luv, kmf=sc_mf, threshold=1.0E-5)

        start_time = time.time()
        sc_nnz, sc_nnk = sc_helper.get_total_unique_terms_above_thresh(return_nk_counter=True)
        end_time = time.time()
        print("nnz calculation sc: ", end_time - start_time)
        sc_res = cost_sparse(sc_nso, 1, 1, sc_nnz, 1.0E-3, 10, 20_000)
        sc_tof_per_step.append(sc_res[0])
        sc_Nk_vals.append(Nk)
        sc_N_vals.append(nso)
        sc_nnz_per_nk.append(sc_nnz / sc_nnk)
        sc_nnz_vals.append(sc_nnz)

    tof_per_step = np.array(tof_per_step)
    sc_tof_per_step = np.array(sc_tof_per_step)
    Nk_vals = np.array(Nk_vals)
    N_vals = np.array(N_vals)
    sc_Nk_vals = np.array(sc_Nk_vals)
    sc_N_vals = np.array(sc_N_vals)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sc_y_vals = sc_nnz_vals / np.array(sc_Nk_vals)**4
    sc_x_vals = np.array(sc_Nk_vals) # * np.array(sc_N_vals)
    m, b = fit_linear_log_log(sc_x_vals, sc_y_vals)
    ax.loglog(sc_x_vals, sc_y_vals,  
              marker='o', mfc=colors[1], mec=colors[1],  linestyle='',
              color=colors[1], alpha=0.5)
    X = np.logspace(np.log(np.min(sc_x_vals)), np.log(np.max(sc_x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, '--', alpha=1., color=colors[1], label=r'SC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    y_vals = nnz_vals / np.array(nnk_vals)
    x_vals = Nk_vals#  * np.array(N_vals)
    print(x_vals)
    print(y_vals)
    m, b = fit_linear_log_log(x_vals, y_vals)
    ax.loglog(x_vals, y_vals,  
              marker='o', mfc=colors[0], mec=colors[0],  linestyle='',
              color=colors[0], alpha=0.5)

    X = np.logspace(np.log(np.min(x_vals)), np.log(np.max(x_vals)), 10, base=np.e)
    Y = np.exp(b) * X**m
    ax.loglog(X, Y, '-', color=colors[0], alpha=1., label=r'UC: $\mathcal{{O}}(N_{{k}}^{{{:1.3f}}})$'.format(m))

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{k}$", fontsize=14)
    ax.set_ylabel(r"NNZ / $f(N_{k})$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14,ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("carbon_sparse_nk_scaling.png", format='PNG', dpi=300)
    plt.savefig("carbon_sparse_nk_scaling.pdf", format='PDF', dpi=300)



if __name__ == "__main__":
    # idealized_overcounting()
    # idealized_n_overcounting()
    # true_counting_routine()
    # true_nk_counting_routine()
    # true_N_counting_routine()
    # carbon_counting_routine()
    carbon_counting_nnz_per_nk_routine()
