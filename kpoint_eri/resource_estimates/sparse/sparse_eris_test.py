from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

def test_count_number_of_non_zero_elements():
    # kmf = utils.build_test_system_diamond('gth-dzvp')
    cell, kmf = utils.init_from_chkfile('diamond_221.chk')
    num_non_zero = sparse.count_number_of_non_zero_elements(kmf)
    nk = 4
    nmo = 26
    no_thresh = nk**3 * nmo**4
    print(num_non_zero, no_thresh)
