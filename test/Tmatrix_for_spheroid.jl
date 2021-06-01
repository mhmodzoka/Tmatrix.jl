using Tmatrix

rx, rz = 1e-6, 1.3e-6
n_max = 3
k1 = 1e7
k2 = 1.5e7 + 1e3 * im

@time T = calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2)

rx, rz = BigFloat(1e-6), BigFloat(1.3e-6)
n_max = 3
k1 = BigFloat(1e7) + 0*im
k2 = BigFloat(1.5e7) + BigFloat(1e3) * im
@time T = calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2)
