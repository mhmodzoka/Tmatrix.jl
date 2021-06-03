using Traceur
using Tmatrix

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 3
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3

# Using Complex numbers
@trace T_complex = calculate_Tmatrix_for_spheroid(rx, rz, n_max, Complex(k1_r, k1_i), Complex(k2_r, k2_i))
