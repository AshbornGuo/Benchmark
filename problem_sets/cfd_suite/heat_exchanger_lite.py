import numpy as np

def heatexchanger_bounds_from_rules(
    n_rows=3,
    n_coeffs_num=4,
    n_betas_each_row=None,
    n_coeffs_radii_each_row=None,
    # These defaults are exactly the ones in the original source you pasted:
    # PipeInterface(... clb=-1, cub=1)
    clb=-1.0, cub=1.0,
    # PipeRow(... alb=0, aub=10, blb=0, bub=10)
    alb=0.0, aub=10.0,
    blb=0.0, bub=10.0,
    # MonotonicBetaCDF(... wlb=0, wub=1)
    wlb=0.0, wub=1.0,
):
    """
    Build the 28-D (or generally D-D) lower/upper bounds following the
    original Exeter_CFD_Problems rules in PipeInterface.get_decision_boundary().

    Decision vector order (exactly as PipeInterface.get_decision_boundary()):
      [ num_pipes_cheb_coeffs,
        row1: alphas, betas, omegas, radii_cheb_coeffs,
        row2: alphas, betas, omegas, radii_cheb_coeffs,
        row3: alphas, betas, omegas, radii_cheb_coeffs ]
    """
    if n_betas_each_row is None:
        n_betas_each_row = [2] * n_rows
    if n_coeffs_radii_each_row is None:
        n_coeffs_radii_each_row = [2] * n_rows

    assert len(n_betas_each_row) == n_rows
    assert len(n_coeffs_radii_each_row) == n_rows

    lb = []
    ub = []

    # 1) number-of-pipes Chebyshev coeffs
    lb.extend([clb] * int(n_coeffs_num))
    ub.extend([cub] * int(n_coeffs_num))

    # 2) each row: alphas, betas, omegas, radii cheb coeffs
    for i in range(n_rows):
        nb = int(n_betas_each_row[i])
        nr = int(n_coeffs_radii_each_row[i])

        # alphas
        lb.extend([alb] * nb)
        ub.extend([aub] * nb)

        # betas
        lb.extend([blb] * nb)
        ub.extend([bub] * nb)

        # omegas
        lb.extend([wlb] * nb)
        ub.extend([wub] * nb)

        # radii Chebyshev coeffs
        lb.extend([clb] * nr)
        ub.extend([cub] * nr)

    return np.array(lb, dtype=float), np.array(ub, dtype=float)


if __name__ == "__main__":
    lb, ub = heatexchanger_bounds_from_rules(
        n_rows=3,
        n_coeffs_num=4,
        n_betas_each_row=[2, 2, 2],
        n_coeffs_radii_each_row=[2, 2, 2],
    )
    print("D =", lb.shape[0])
    print("lb =", lb)
    print("ub =", ub)
