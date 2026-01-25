import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


def test_sparse_vs_bincount_equivalence():
    # Synthetic small active set
    rng = np.random.RandomState(1)
    n = 20
    pos = rng.uniform(0, 100, size=(n, 2))
    radius = 20.0

    kd = cKDTree(pos)
    dist_mat = kd.sparse_distance_matrix(kd, radius, output_type='coo_matrix')

    rows = dist_mat.row
    cols = dist_mat.col
    dists = dist_mat.data
    mask_pairs = rows < cols
    rows = rows[mask_pairs]
    cols = cols[mask_pairs]
    dists = dists[mask_pairs]

    # Map to i,j indices (0..n-1)
    idx_i = rows
    idx_j = cols

    xi = pos[idx_i, 0]
    yi = pos[idx_i, 1]
    xj = pos[idx_j, 0]
    yj = pos[idx_j, 1]

    dx = xj - xi
    dy = yj - yi
    dist = dists + 1e-6

    # simple weights: inverse distance
    p = 1.0 / dist

    ux_contrib_i = (dx / dist) * p
    uy_contrib_i = (dy / dist) * p
    ux_contrib_j = -(dx / dist) * p
    uy_contrib_j = -(dy / dist) * p

    # Bincount results
    ux_b = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([ux_contrib_i, ux_contrib_j]), minlength=n)
    uy_b = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([uy_contrib_i, uy_contrib_j]), minlength=n)

    # Sparse matmul approach
    ncols = len(idx_i)
    col_idx = np.concatenate([np.arange(ncols), np.arange(ncols)])
    row_idx = np.concatenate([idx_i, idx_j])

    data_ux = np.concatenate([ux_contrib_i, ux_contrib_j])
    Mux = coo_matrix((data_ux, (row_idx, col_idx)), shape=(n, ncols))
    ux_s = np.asarray(Mux.dot(np.ones(ncols, dtype=np.float64)), dtype=np.float64).ravel()

    data_uy = np.concatenate([uy_contrib_i, uy_contrib_j])
    Muy = coo_matrix((data_uy, (row_idx, col_idx)), shape=(n, ncols))
    uy_s = np.asarray(Muy.dot(np.ones(ncols, dtype=np.float64)), dtype=np.float64).ravel()

    # Compare
    assert np.allclose(ux_b, ux_s)
    assert np.allclose(uy_b, uy_s)
