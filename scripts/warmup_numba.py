import numpy as np
from cenop.optimizations import numba_helpers as nh

count = 10
idx_i = np.array([0,1,2], dtype=np.int64)
idx_j = np.array([3,4,5], dtype=np.int64)
ux_ci = np.random.randn(len(idx_i)).astype(np.float64)
uy_ci = np.random.randn(len(idx_i)).astype(np.float64)
ux_cj = np.random.randn(len(idx_i)).astype(np.float64)
uy_cj = np.random.randn(len(idx_i)).astype(np.float64)
p_i = np.random.rand(len(idx_i)).astype(np.float64)
p_j = np.random.rand(len(idx_i)).astype(np.float64)
ux_total = np.zeros(count, dtype=np.float64)
uy_total = np.zeros(count, dtype=np.float64)
sw_total = np.zeros(count, dtype=np.float64)
nh.accumulate_social_totals(count, idx_i, idx_j, ux_ci, uy_ci, ux_cj, uy_cj, p_i, p_j, ux_total, uy_total, sw_total)
print('Warm-up complete')
