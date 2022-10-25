import numpy as np

from pyscf.pbc import gto, tools, df
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.dft import gen_grid, numint

def supercell_isdf(mydf: df.FFTDF, interp_indx: np.ndarray, kpoint=np.zeros(3),
                   orbitals=None, grid_points=None):
    r"""
    Build ISDF-THC representation of ERI tensors.

    (pq|rs) = \sum_{mn} chi_{pm}^* chi_{qm} zeta_{mn} chi_{rn}^* chi_{sn}

    :returns tuple: (zeta, chi, Theta): central tensor, orbitals on interpolating
        points and matrix interpolating vectors Theta = [xi_1, xi_2.., xi_Nmu].
        Note chi is not necessarily normalized (check).
    """

    cell = mydf.cell
    if grid_points is None:
        grid_points = cell.gen_uniform_grids(mydf.mesh)
    if orbitals is None:
        orbitals = numint.eval_ao(cell, grid_points)

    num_grid_points = len(grid_points)
    num_interp_points = len(interp_indx)
    # orbitals on interpolating grid.
    interp_orbitals = orbitals[interp_indx]
    # Form pseudo-densities
    # P[R, r_mu] = \sum_{i} phi_{R,i}^* phi_{mu,i}
    pseudo_density = np.einsum(
        "Ri,mi->Rm", orbitals.conj(), interp_orbitals, optimize=True
    )
    # [Z C^]_{J, mu} = (sum_i phi_{i, J}^* phi_{i, mu}) (sum_j phi_{j, J} # phi_{i, mu})
    ZC_dag = np.einsum(
        "Rm,Rm->Rm", pseudo_density, pseudo_density.conj(), optimize=True
    )
    # Just down sample from ZC_dag
    CC_dag = ZC_dag[interp_indx].copy()
    # Solve ZC_dag = Theta CC_dag
    # Theta =  ZC_dag (CC_dag)^{-1}
    CC_dag_inv = np.linalg.pinv(CC_dag)
    Theta = ZC_dag @ CC_dag_inv

    # FFT Theta[R, mu] -> Theta[mu, G]
    # Transpose as fft expects contiguous.
    Theta_G = tools.fft(Theta.T, mydf.mesh)
    coulG = (
        tools.get_coulG(cell, k=kpoint, mesh=mydf.mesh)
    )
    weighted_coulG = coulG * cell.vol / num_grid_points**2.0

    # zeta_{mu,nu} = \sum_G 4pi/(omega * G^2) zeta_{mu,G} * (zeta_G*){nu, G}
    Theta_G_tilde = np.einsum("iG,G->iG", Theta_G, weighted_coulG)
    central_tensor = (Theta_G_tilde) @ Theta_G.conj().T
    return interp_orbitals, central_tensor, Theta


class SupercellISDFHelper(object):
    """
    """

    def __init__(self, zeta, interp_orbitals_mo):
        self.zeta = zeta
        self.interp_orbitals_mo = interp_orbitals_mo

    def get_eri(self, mo_coeff):
        # interp_orbs_mo = np.einsum("pi,pm->im", mo_coeff, self.interp_orbitals, optimize=True)
        Lijn = np.einsum(
            "im,jm,mn->ijn",
            self.interp_orbitals_mo.conj(),
            self.interp_orbitals_mo,
            self.zeta,
            optimize=True,
        )
        eris = np.einsum(
            "ijn,kn,ln->ijkl",
            Lijn,
            self.interp_orbitals_mo.conj(),
            self.interp_orbitals_mo,
            optimize=True,
        )
        return eris
