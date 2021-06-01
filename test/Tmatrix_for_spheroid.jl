using Tmatrix

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 3
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3

# Using Complex numbers
@time T_complex = calculate_Tmatrix_for_spheroid(rx, rz, n_max, Complex(k1_r, k1_i), Complex(k2_r, k2_i))

# Using Complex numbers - BigFloat
@time T_complex_BigFloat = calculate_Tmatrix_for_spheroid(BigFloat(rx), BigFloat(rz), n_max, Complex(BigFloat(k1_r), BigFloat(k1_i)), Complex(BigFloat(k2_r), BigFloat(k2_i)), rotationally_symmetric=true)

# Using Real numbers
@time T_Real = Tmatrix.calculate_Tmatrix_for_spheroid_SeparateRealImag(rx, rz, n_max, k1_r, k1_i, k2_r, k2_i)

# Using Real numbers - Big Float
@time T_Real_BigFloat = Tmatrix.calculate_Tmatrix_for_spheroid_SeparateRealImag(BigFloat(rx), BigFloat(rz), n_max, BigFloat(k1_r), BigFloat(k1_i), BigFloat(k2_r), BigFloat(k2_i))
