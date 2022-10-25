import numpy as np

from pyscf.pbc import gto, tools, df
from pyscf.pbc.df.aft import weighted_coulG
from pyscf.pbc.dft import gen_grid, numint

def supercell_isdf(mydf: df.FFTDF, interp_indx: np.ndarray, kpoint=np.zeros(3),
                   orbitals=None):
    r"""
    Build ISDF-THC representation of ERI tensors.

    (pq|rs) = \sum_{mn} chi_{pm}^* chi_{qm} zeta_{mn} chi_{rn}^* chi_{sn}

    :returns tuple: (zeta, chi, Theta): central tensor, orbitals on interpolating
        points and matrix interpolating vectors Theta = [xi_1, xi_2.., xi_Nmu]
    """

    cell = mydf.cell
    grid_points = cell.gen_uniform_grids(mydf.mesh)
    if orbitals is None:
        orbitals = numint.eval_ao(cell, grid_points)

    num_grid_points = len(grid_points)
    num_interp_points = len(interp_indx)
    # orbitals on interpolating grid.
    interp_orbitals = orbitals[interp_indx]
    # Form pseudo-densities
    # P[r_J, r_mu] = \sum_{i} phi_{i, J}^* phi_{i, mu}
    pseudo_density = np.einsum(
        "iJ,im->Jm", orbitals.conj(), interp_orbitals, optimize=True
    )
    # [Z C^]_{J, mu} = (sum_i phi_{i, J}^* phi_{i, mu}) (sum_j phi_{j, J} # phi_{i, mu})
    ZC_dag = np.einsum(
        "Jm,Jm->Jm", pseudo_density, pseudo_density.conj(), optimize=True
    )
    # Just down sample from ZC_dag
    CC_dag = ZC_dag[interp_indx].copy()
    # Solve ZC_dag = Theta CC_dag
    # Theta = (CC_dag)^{-1} ZC_dag
    CC_dag_inv = np.linalg.pinv(CC_dag)
    Theta = CC_dag_inv @ ZC_dag

    # FFT Theta[mu, r] -> Theta[mu, G]
    Theta_G = tools.fft(Theta, mydf.mesh)
    coulG = (
        tools.get_coulG(cell, k=kpoint, gs=mydf.mesh)
    )
    weighted_coulG = coulG * cell.vol / num_grid_points

    # zeta_{mu,nu} = \sum_G 4pi/(omega * G^2) zeta_{mu,G} * (zeta_G*){nu, G}
    central_tensor = (Theta_G * weighted_coulG) @ Theta_G.conj().T
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
