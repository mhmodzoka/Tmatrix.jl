using Tmatrix

import Zygote

n_θ_points = 10
n_ϕ_points = 2
m, n, m_, n_ = 2, 3, 2, 5
n_max = 1
k1_r, k1_i, k2_r, k2_i = 1e5, 1e3, 2e5, 3e3
rotationally_symmetric = true
symmetric_about_plan_perpendicular_z = true
rx, rz = 1e-6, 1.2e-6
kind = "regular"
J_superscript = 11
Q_superscript = J_superscript

θ_1D_array = LinRange(1e-16, π, n_θ_points);
ϕ_1D_array = LinRange(1e-16, 2π, n_ϕ_points);
if rotationally_symmetric        
    θ_array = collect(θ_1D_array)
    ϕ_array = zeros(size(θ_array))
else
    # eventually, this should be removed. I just keep it for sanity checks.
    θ_array, ϕ_array = Tmatrix.meshgrid(θ_1D_array, ϕ_1D_array);
end
r_array, n̂_array = Tmatrix.ellipsoid(rx, rz, θ_array);

# test "J_mn_m_n__integrand_SeparateRealImag"
J = Tmatrix.J_mn_m_n__integrand_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript)
J_ = Tmatrix.J_mn_m_n__integrand(m, n, m_, n_, complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(J))
println("Displaying result from complex function ------------------------------")
display(J_)

Zygote.jacobian(Tmatrix.J_mn_m_n__integrand_SeparateRealImag, m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array,kind,J_superscript)


# test "J_mn_m_n__SeparateRealImag"
J = Tmatrix.J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript, rotationally_symmetric)
J_ = Tmatrix.J_mn_m_n_(m, n, m_, n_, complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript, rotationally_symmetric=rotationally_symmetric)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(J))
println("Displaying result from complex function ------------------------------")
display(J_)

# the problem that Zygote is trying to differentiate "rotationally_symmetric".
# how can I prevent Zygote from differentiating with respect to a given argument?
Zygote.jacobian(
    (m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Tmatrix.J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript, rotationally_symmetric),
    m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


# test "Q_mn_m_n_SeparateRealImag"
@time Q = Tmatrix.Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time Q_ = Tmatrix.Q_mn_m_n_(m, n, m_, n_, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, Q_superscript=Q_superscript, rotationally_symmetric=rotationally_symmetric)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

Zygote.jacobian(
    (m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Tmatrix.Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plan_perpendicular_z),
    m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


# test "Q_matrix_SeparateRealImag"
@time Q = Tmatrix.Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time Q_ = Tmatrix.Q_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array, kind=kind, rotationally_symmetric=rotationally_symmetric, symmetric_about_plan_perpendicular_z=false, verbose=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

# The jacobian calculation is so slow. For n_max = 1, it takes 52.409485 seconds. This is slow compared to 0.015419 seconds required to evaluate the function
Zygote.jacobian(
    (n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Tmatrix.Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plan_perpendicular_z),
     n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


# test "T_matrix_SeparateRealImag"
@time T = Tmatrix.T_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time T_ = Tmatrix.T_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; rotationally_symmetric=rotationally_symmetric, symmetric_about_plan_perpendicular_z=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(T))
println("Displaying result from complex function ------------------------------")
display(T_)

Zygote.jacobian(Tmatrix.T_matrix_SeparateRealImag, n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
