from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

def test_lambda_sparse():
    cell, kmf = utils.init_from_chkfile('diamond_221.chk')
    lambda_tot, lambda_T, lambda_V  = sparse.compute_lambda(kmf)
    print(lambda_tot)
