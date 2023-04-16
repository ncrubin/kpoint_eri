import numpy as np

def QR3(L, M1):
    r"""
    QR[Ll_, m_] := Ceiling[MinValue[{Ll/2^k + m*(2^k - 1), k >= 0}, k \[Element] Integers]];
    """
    k = 0.5 * np.log2(L / M1)
    value = lambda k: L / np.power(2, k) + M1 * (np.power(2, k) - 1)
    try:
        assert k >= 0
    except AssertionError:
        k_opt = 0
        val_opt = np.ceil(value(k_opt))
        assert val_opt.is_integer()
        return int(k_opt), int(val_opt)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)


def QR2(L1, L2, M):
    """
     Table[Ceiling[L1/2^k1]*Ceiling[L2/2^k2] + M*(2^(k1 + k2) - 1), {k1, 1,
    10}, {k2, 1, 10}]
    """
    min_val = np.inf
    for k1 in range(1, 11):
        for k2 in range(1, 11):
            test_val = np.ceil(L1 / (2**k1)) * np.ceil(L2 / (2**k2)) + M * (
                2 ** (k1 + k2) - 1
            )
            if test_val < min_val:
                min_val = test_val
    return int(min_val)


def QI2(L1, Lv2):
    """
    QI2[L1_, L2_] :=
    Min[Table[
    Ceiling[L1/2^k1]*Ceiling[L2/2^k2] + 2^(k1 + k2), {k1, 1, 10}, {k2,
      1, 10}]];
    """
    min_val = np.inf
    for k1 in range(1, 11):
        for k2 in range(1, 11):
            test_val = np.ceil(L1 / (2**k1)) * np.ceil(Lv2 / (2**k2)) + 2 ** (
                k1 + k2
            )
            if test_val < min_val:
                min_val = test_val
    return int(min_val)

