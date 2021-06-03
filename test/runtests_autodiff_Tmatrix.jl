using Tmatrix

import Zygote

n_θ_points = 10
n_ϕ_points = 2
m, n, m_, n_ = 2, 3, 2, 5
n_max = 1
k1_r, k1_i, k2_r, k2_i = 1e5, 1e3, 2e5, 3e3
rotationally_symmetric = false
symmetric_about_plane_perpendicular_z = false
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
@time Q = Tmatrix.Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plane_perpendicular_z)
@time Q_ = Tmatrix.Q_mn_m_n_(m, n, m_, n_, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, Q_superscript=Q_superscript, rotationally_symmetric=rotationally_symmetric)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

Zygote.jacobian(
    (m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Tmatrix.Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plane_perpendicular_z),
    m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


# test "Q_matrix_SeparateRealImag"
@time Q = Tmatrix.Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plane_perpendicular_z)
@time Q_ = Tmatrix.Q_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array, kind=kind, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=false, verbose=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

# The jacobian calculation is so slow. For n_max = 1, it takes 52.409485 seconds. This is slow compared to 0.015419 seconds required to evaluate the function
Zygote.jacobian(
    (n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Tmatrix.Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plane_perpendicular_z),
     n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


# test "T_matrix_SeparateRealImag"
@time T = Tmatrix.T_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plane_perpendicular_z)
@time T_ = Tmatrix.T_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(Tmatrix.get_complex_matrix_from_concatenated_real_imag(T))
println("Displaying result from complex function ------------------------------")
display(T_)

Zygote.jacobian(Tmatrix.T_matrix_SeparateRealImag, n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plane_perpendicular_z)



import Tmatrix.J_mn_m_n__integrand_SeparateRealImag
import Tmatrix.surface_integrand

using VectorSphericalWaves
using ComplexOperations
function J_mn_m_n__integrand_SeparateRealImag(
        m::Int, n::Int, m_::Int, n_::Int,
        k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}, ϕ_array::AbstractVecOrMat{R}, n̂_array::Any,
        kind::String, J_superscript::Int
    ) where {R <: Real}

    # TODO: do you think I should have done these multiplication outside, rather than repeating them whenever the function is called? or Julia should eliminate these duplicate calculation?
    kr1_r = k1_r .* r_array
    kr1_i = k1_i .* r_array
    kr2_r = k2_r .* r_array
    kr2_i = k2_i .* r_array

    # calculate the integrand
    if J_superscript == 11 # TODO: this if-statement can be done more nicely. We separate J_superscript into two pieces, the number 1 represents M_mn_wave_SeparateRealImag, while number 2 represents N_mn_wave_SeparateRealImag        
        first_function = M_mn_wave_SeparateRealImag
        second_function = M_mn_wave_SeparateRealImag
    elseif J_superscript == 12
        first_function = M_mn_wave_SeparateRealImag
        second_function = N_mn_wave_SeparateRealImag
    elseif J_superscript == 21
        first_function = N_mn_wave_SeparateRealImag
        second_function = M_mn_wave_SeparateRealImag
    elseif J_superscript == 22
        first_function = N_mn_wave_SeparateRealImag
        second_function = N_mn_wave_SeparateRealImag
    else
        throw(DomainError("J_superscript must be any of [11,12,21,22]"))
    end
    
    kind_first_function = "regular"
    if kind == "irregular"                
        kind_second_function = "irregular"
    elseif kind == "regular"        
        kind_second_function = "regular"
    else
        throw(DomainError("""kind must be any of ["regular", "irregular"]"""))
    end

    # the cross product    
    cross_product_MN = complex_vector_cross_product.(
        first_function.(m_, n_, kr2_r, kr2_i, θ_array, ϕ_array, kind_first_function),
        second_function.(-m, n, kr1_r, kr1_i, θ_array, ϕ_array, kind_second_function),
    )
    println("size of cross_product_MN : $(size(cross_product_MN))")
    println(cross_product_MN)

    println("size of n̂_array : $(size(n̂_array))")
    println(n̂_array)

    cross_product_MN_dot_n̂ = (-1).^m .* complex_vector_dot_product.(cross_product_MN, n̂_array)

    println("size of cross_product_MN_dot_n̂ : $(size(cross_product_MN_dot_n̂))")
    println(cross_product_MN_dot_n̂)

    println("size of cross_product_MN_dot_n̂[1,1] : $(size(cross_product_MN_dot_n̂[1,1]))")
    println(cross_product_MN_dot_n̂[1,1])

    println("size of r_array : $(size(r_array))")
    println(r_array)

    println("size of θ_array : $(size(θ_array))")
    println(θ_array)
    
    J_integrand = Tmatrix.surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    println("size of J_integrand : $(size(J_integrand))")
    println(J_integrand)

    # return vcat(J_integrand...) # I had to flatten all nested arrays.
    return hcat([i[1] for i in J_integrand], [i[2] for i in J_integrand])
end 


"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
        integrand::AbstractVecOrMat{C}, r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}
    ) where {R <: Real, C <: Complex{R}} 
    return integrand .* r_array.^2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
        integrand::AbstractVecOrMat{R}, r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}
    ) where {R <: Real, C <: Complex{R}} 
    return integrand .* r_array.^2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
        integrand::AbstractVecOrMat{AbstractVecOrMat{C}}, r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}
    ) where {R <: Real, C <: Complex{R}} 
    return integrand .* r_array.^2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
        integrand::AbstractVecOrMat{AbstractVecOrMat{R}}, r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}
    ) where {R <: Real, C <: Complex{R}} 
    return integrand .* r_array.^2 .* sin.(θ_array)
end