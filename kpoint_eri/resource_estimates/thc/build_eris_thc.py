import numpy as np
from itertools import product

def build_gamma_eris(etapP, MPQ):
    eris = np.einsum(
            'pP,qP,PQ,rQ,sQ->pqrs',
            etapP.conj(),
            etapP,
            MPQ,
            etapP.conj(),
            etapP,
            optimize=True)
