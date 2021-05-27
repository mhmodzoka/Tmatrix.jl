# %%
using Tmatrix
using Profile

rx, rz = 1e-6, 1.3e-6
n_max = 3
k1 = 1e7
k2 = 1.5e7 + 1e3 * im

@profview calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2)
