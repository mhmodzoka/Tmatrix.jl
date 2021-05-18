# NOTE: Lessons learned:
# 1- All functions should return only one Array, which contains no nested arrays. This can be done by vcat(array...)

# TODO:
# 1- Find a way to enforce concrrete types at the T_matrix function. So far, I have defined abstract types as inputs for all functions. I am not sure how to convert to a concrete type mathcing my computation requirement (e.g., BigFloats)
# 2- Check why trapz_ELZOUKA is not returning exactly the same result as Trapz.trapz
# 3- The functions I have wrote separating real and imaginary parts are slower, figure out why

using Tmatrix
using ComplexOperations
using LinearAlgebra
using VectorSphericalWaves
using Trapz

import Zygote
#######################
# TOBE moved to "ComplexOperations" package
"""
    Claculate dot product of two vectors
Each is represented as 3x2 or 3x1 Array. The first and second columns represent the real and imag parts, respectively
For real vectors, we can input them as 3x1 Array or 3-element Vector
"""
function complex_vector_dot_product(A, B)    
    if size(A, 2) == 1; A = hcat(A, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    if size(B, 2) == 1; B = hcat(B, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    return complex_vector_dot_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_dot_product(A_r, A_i, B_r, B_i)
    return (
    complex_multiply(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate cross product of two vectors
Each is represented as 3x2 or 3x1 Array. The first and second columns represent the real and imag parts, respectively
For real vectors, we can input them as 3x1 Array or 3-element Vector
"""
function complex_vector_cross_product(A, B)
    if size(A, 2) == 1; A = hcat(A, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    if size(B, 2) == 1; B = hcat(B, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    return complex_vector_cross_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_cross_product(A_r, A_i, B_r, B_i)
    vcat(
    complex_multiply(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

"""
    Calculate the inverse of a complex matrix A+iB, where A and B are real matrices

function complex_matrix_inversion(A, B)
    inv_A_B = inv(A) * B
    C = inv(A + B * inv_A_B)
    D = -inv_A_B * C
    return C, D
end
"""
function complex_matrix_inversion(A, B)    
    C = inv(A + B * inv(A) * B)
    D = -1 .* inv(A) * B * C
    return C, D
end

"""
    Multiply two matrices A+iB and C+iD, where A,B,C,D are real matrices
"""
function complex_matrix_multiplication(A, B, C, D)
    return (A * C - B * D, A * D + B * C)
end

A = rand(3, 3); B = rand(3, 3)
@time C, D = complex_matrix_inversion(A, B)
@time inv(A + im * B)

A_r = [1,2,3]; A_i = [1, 20, 30]
B_r = [10,2,30]; B_i = [1, 2, 3]
AB = complex_vector_cross_product(A_r, A_i, B_r, B_i)
AB_ = complex_vector_cross_product(hcat(A_r, A_i), hcat(B_r, B_i))
AB__ = cross(A_r + im * A_i, B_r + im * B_i)

AB = complex_vector_dot_product(A_r, A_i, B_r, B_i)
AB_ = complex_vector_dot_product(hcat(A_r, A_i), hcat(B_r, B_i))
AB__ = Tmatrix.vector_dot_product(A_r + im * A_i, B_r + im * B_i)

# TOBE moved to "ComplexOperations" package
#######################


#######################
# TOBE moved to a new package for trapezoidal integraion, that will be compatible with autodiff
"""
    numerical integral using trapezoidal rule
x and y are 1D arrays
"""
function trapz_ELZOUKA(x, y)
    # TODO: small error if compared with Trapz.trapz
    base = x[2:end] - x[1:end - 1]
    av_height = (y[2:end] + y[1:end - 1]) / 2
    areas = base .* av_height
    total_area = sum(areas)
    return total_area
end
#######################


#######################
# TOBE moved to Tmatrix module
# getting indices of T-matrix
"""
    Get 2D array, first and second columns are n and m values, respectively.
m = -n : 1 : +n
n will be the repeated value of input
"""
function get_n_m_array_given_n(n)
    return hcat(repeat([n], 2 * n + 1), -n:n)
end

"""
    Get 2D array, first and second columns are n and m values, respectively.
This is looping `get_n_m_array_given_n` over all n = 1:1:n_max
"""
function get_n_m_array_given_n_max(n_max)
    return vcat(get_n_m_array_given_n.(1:n_max)...)
end

"""
    Get 2D array, first, second and third columns are n, m, and idx values, respectively.
The idx value is just the index of the row.
"""
function get_n_m_idx_array_given_n_max(n_max)
    nm_array = get_n_m_array_given_n_max(n_max)
    return hcat(nm_array, 1:size(nm_array, 1))
end

"""
    Get 4 matrices for m, n, m_, n_, corresponding to rank and order of incident and scattered VSWF, represented by elements of T-matrix
"""
function get_m_n_m__n__matrices_for_T_matrix(n_max)
    n_m_idx = get_n_m_idx_array_given_n_max(n_max)
    n_array = n_m_idx[:,1]
    m_array = n_m_idx[:,2]
    idx_array = n_m_idx[:,3]
    
    idx_max = idx_array[end]
    idx_matrix = repeat(idx_array, 1, idx_max)
    idx__matrix = repeat(idx_array', idx_max, 1)

    n_matrix = n_array[idx_matrix]
    m_matrix = m_array[idx_matrix]

    n__matrix = n_array[idx__matrix]
    m__matrix = m_array[idx__matrix]

    return m_matrix, n_matrix, m__matrix, n__matrix
end

"""
    using for loops, Get 4 matrices for m, n, m_, n_, corresponding to rank and order of incident and scattered VSWF, represented by elements of T-matrix
"""
function get_m_n_m__n__matrices_for_T_matrix_with_forloop(n_max)
    idx_max = Tmatrix.get_max_single_index_from_n_max(n_max)
    n_matrix = zeros(Int, idx_max, idx_max)
    m_matrix = zeros(Int, idx_max, idx_max)
    n__matrix = zeros(Int, idx_max, idx_max)
    m__matrix = zeros(Int, idx_max, idx_max)
    idx = 0;
    for n = 1:n_max
        for m = -n:n            
            idx += 1            
            idx_ = 0;
            for n_ = 1:n_max
                for m_ = -n_:n_                    
                    idx_ += 1
                    m_matrix[idx, idx_] = m
                    n_matrix[idx, idx_] = n
                    m__matrix[idx, idx_] = m_
                    n__matrix[idx, idx_] = n_
                end
            end
        end
    end
    return m_matrix, n_matrix, m__matrix, n__matrix
end

"""
    Create m,n,m_,n_ matrices with for loop, and compare the result with `get_m_n_m__n__matrices_for_T_matrix`
"""
function validate_get_m_n_m__n__matrices_for_T_matrix(n_max)
    
    m_matrix, n_matrix, m__matrix, n__matrix = get_m_n_m__n__matrices_for_T_matrix_with_forloop(n_max)
    _m_matrix, _n_matrix, _m__matrix, _n__matrix = get_m_n_m__n__matrices_for_T_matrix(n_max)

    println(_n_matrix == n_matrix)
    println(_m_matrix == m_matrix)
    println(_n__matrix == n__matrix)
    println(_m__matrix == m__matrix)
end
#######################
"""
    Separate real and imaginary parts. I assume that A is a hcat of real and imag parts
"""
function separate_real_imag(A)
    return A[:,1:Int(end / 2)], A[:,Int(end / 2) + 1:end]
end

"""
    return the complex matrix, given matrix A which is hcat of real and imag parts
"""
function get_complex_matrix_from_concatenated_real_imag(A)
    A_r, A_i = separate_real_imag(A)
    return A_r + im * A_i
end

n_θ_points = 10
n_ϕ_points = 2
m, n, m_, n_ = 2, 3, 2, 5
n_max = 1
k1_r, k1_i, k2_r, k2_i = 1e5, 1e3, 2e5, 3e3
rotationally_symmetric = true
symmetric_about_plan_perpendicular_z = true
use_Alok_vector_preallocation = true
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
r_array, n̂_array = Tmatrix.ellipsoid(rx, rz, θ_array; use_Alok_vector_preallocation=use_Alok_vector_preallocation);




function J_mn_m_n__integrand_SeparateRealImag(
    m::Int, n::Int, m_::Int, n_::Int,
    k1_r::Real, k1_i::Real, k2_r::Real, k2_i::Real,
    r_array::AbstractArray, θ_array::AbstractArray, ϕ_array::AbstractArray, n̂_array::AbstractArray,
    kind="regular", J_superscript=11
    )

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
    cross_product_MN_dot_n̂ = (-1).^m .* complex_vector_dot_product.(cross_product_MN, n̂_array)

    J_integrand = Tmatrix.surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    return vcat(J_integrand...) # I had to flatten all nested arrays.
end    


J = J_mn_m_n__integrand_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript)
J_ = Tmatrix.J_mn_m_n__integrand(m, n, m_, n_, complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript)
println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(get_complex_matrix_from_concatenated_real_imag(J))
println("Displaying result from complex function ------------------------------")
display(J_)

Zygote.jacobian(J_mn_m_n__integrand_SeparateRealImag, m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array,kind,J_superscript)


function J_mn_m_n__SeparateRealImag(
    m::Int, n::Int, m_::Int, n_::Int,
    k1_r::Real, k1_i::Real, k2_r::Real, k2_i::Real,
    r_array::AbstractArray, θ_array::AbstractArray, ϕ_array::AbstractArray, n̂_array::AbstractArray,
    kind, J_superscript, rotationally_symmetric,
    )
    if rotationally_symmetric
        # make sure that θ_array is 1D
        if length(size(θ_array)) != 1
            throw(DomainError("Since you have indicated << rotationally_symmetric = true >>, θ_array has to be 1D. Now it is $(length(size(θ_array)))D"))
        end
        ϕ_array = zeros(size(θ_array))
    end
    
    if rotationally_symmetric && (m != m_)
        # the integral over ϕ is 2π * δ_m_m_, so it is zero if m != m_
        J_integrand_dS_r = zeros(size(θ_array))
        J_integrand_dS_i = zeros(size(θ_array))
        return hcat(0, 0)
    
    else
        J_integrand_dS = J_mn_m_n__integrand_SeparateRealImag(
            m, n, m_, n_,
            k1_r, k1_i, k2_r, k2_i,
            r_array, θ_array, ϕ_array, n̂_array,
            kind, J_superscript,            
        )
        # because J_mn_m_n__integrand_SeparateRealImag returns a flattened J, I have to reshape it to make it work for trapz
        J_integrand_dS_r = reshape(J_integrand_dS[:,1], size(θ_array))
        J_integrand_dS_i = reshape(J_integrand_dS[:,2], size(θ_array))
    end
    
    # surface integral
    if rotationally_symmetric
        # integrate over θ only        
        J_r = 2π * trapz_ELZOUKA((θ_array), J_integrand_dS_r)
        J_i = 2π * trapz_ELZOUKA((θ_array), J_integrand_dS_i)
    else
        # integrate over θ and ϕ
        # TODO: replace this integral with surface mesh quadrature, like this one: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5095443/ , https://github.com/jareeger/Smooth_Closed_Surface_Quadrature_RBF-julia
        # assuming that θ_array, ϕ_array were created with meshgrid function

        # TODO: I need to define trapz_ELZOUKA for 2D integrals. Otherwise, autodiff will not work.
        J_r = trapz((θ_array[:,1], ϕ_array[1,:]), J_integrand_dS_r)
        J_i = trapz((θ_array[:,1], ϕ_array[1,:]), J_integrand_dS_i)
    end

    return hcat(J_r, J_i)    
end

J = J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript, rotationally_symmetric)
J_ = Tmatrix.J_mn_m_n_(m, n, m_, n_, complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript, rotationally_symmetric=rotationally_symmetric)
println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(get_complex_matrix_from_concatenated_real_imag(J))
println("Displaying result from complex function ------------------------------")
display(J_)

# the problem that Zygote is trying to differentiate "rotationally_symmetric".
# how can I prevent Zygote from differentiating with respect to a given argument?
Zygote.jacobian(
    (m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript, rotationally_symmetric),
    m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


function Q_mn_m_n_SeparateRealImag(
    m::Int, n::Int, m_::Int, n_::Int,
    k1_r::Real, k1_i::Real, k2_r::Real, k2_i::Real,
    r_array::AbstractArray, θ_array::AbstractArray, ϕ_array::AbstractArray, n̂_array::AbstractArray, # TODO: how to define a multidimensional array type with Real elements?
    kind, Q_superscript, rotationally_symmetric, symmetric_about_plan_perpendicular_z,    
)
    if Q_superscript == 11; J_superscript_1 = 21 ; J_superscript_2 = 12
    elseif Q_superscript == 12; J_superscript_1 = 11 ; J_superscript_2 = 22
    elseif Q_superscript == 21; J_superscript_1 = 22 ; J_superscript_2 = 11
    elseif Q_superscript == 22; J_superscript_1 = 12 ; J_superscript_2 = 21
    end
    
    if rotationally_symmetric && (m != m_)
        return hcat(0, 0)

    # TODO: adding these lines causes Zygote to fail. Why? The code can work without them.
    # elseif rotationally_symmetric && symmetric_about_plan_perpendicular_z && (m == m_) && (Q_superscript == 11 || Q_superscript == 22) && ! iseven(n + n_)
    #    return hcat(0, 0)

    # elseif rotationally_symmetric && symmetric_about_plan_perpendicular_z && (m == m_) && (Q_superscript == 12 || Q_superscript == 21) && ! isodd(n + n_)
    #    return hcat(0, 0)    
    
    else
        Q_r, Q_i = (
            - 1 .* complex_multiply(
                [0, 1]', complex_multiply(
                    [k1_r, k1_i]', complex_multiply(
                        [k2_r, k2_i]', J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript_1, rotationally_symmetric)
                    )
                )
            )
            
            - 1 .* complex_multiply(
                [0, 1]', complex_multiply(
                    [k1_r, k1_i]', complex_multiply(
                        [k1_r, k1_i]', J_mn_m_n__SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, J_superscript_2, rotationally_symmetric)
                    )
                )
            )
        )    
        return hcat(Q_r, Q_i)    
    end
end


@time Q = Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time Q_ = Tmatrix.Q_mn_m_n_(m, n, m_, n_, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, Q_superscript=Q_superscript, rotationally_symmetric=rotationally_symmetric)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

Zygote.jacobian(
    (m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Q_mn_m_n_SeparateRealImag(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, Q_superscript, rotationally_symmetric, symmetric_about_plan_perpendicular_z),
    m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)




function Q_matrix_SeparateRealImag(
    n_max::Int, k1_r::Real, k1_i::Real, k2_r::Real, k2_i::Real,
    r_array::AbstractArray, θ_array::AbstractArray, ϕ_array::AbstractArray, n̂_array::AbstractArray,
    kind, rotationally_symmetric, symmetric_about_plan_perpendicular_z,
)
    
    m, n, m_, n_ = get_m_n_m__n__matrices_for_T_matrix(n_max)    
    Q_mn_m_n_11 = Q_mn_m_n_SeparateRealImag.(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, [r_array], [θ_array], [ϕ_array], [n̂_array], kind, 11, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
    Q_mn_m_n_12 = Q_mn_m_n_SeparateRealImag.(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, [r_array], [θ_array], [ϕ_array], [n̂_array], kind, 12, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
    Q_mn_m_n_21 = Q_mn_m_n_SeparateRealImag.(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, [r_array], [θ_array], [ϕ_array], [n̂_array], kind, 21, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
    Q_mn_m_n_22 = Q_mn_m_n_SeparateRealImag.(m, n, m_, n_, k1_r, k1_i, k2_r, k2_i, [r_array], [θ_array], [ϕ_array], [n̂_array], kind, 22, rotationally_symmetric, symmetric_about_plan_perpendicular_z)

    Q = vcat(
        (hcat(Q_mn_m_n_11, Q_mn_m_n_12)),
        (hcat(Q_mn_m_n_21, Q_mn_m_n_22))
    )
    get_real_of_nested(x) = x[1] # TODO: is there a better way to separate the first element of a nested array?
    get_imag_of_nested(x) = x[2] # TODO: is there a better way to separate the first element of a nested array?
    Q_r = get_real_of_nested.(Q) # TODO: is there a better way to separate the first element of a nested array?
    Q_i = get_imag_of_nested.(Q) # TODO: is there a better way to separate the first element of a nested array?
    
    # TODO: is it better to return a 2D square matrix, each element is a 2-vector, representing real and imag parts? This will not work with Zygote
    # TODO: or concatenate 2 2D square arrays, the first and second represent the real and imaginary parts?
    return hcat(Q_r, Q_i) 
end


@time Q = Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time Q_ = Tmatrix.Q_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array, kind=kind, rotationally_symmetric=rotationally_symmetric, symmetric_about_plan_perpendicular_z=false, verbose=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(get_complex_matrix_from_concatenated_real_imag(Q))
println("Displaying result from complex function ------------------------------")
display(Q_)

# The jacobian calculation is so slow. For n_max = 1, it takes 52.409485 seconds. This is slow compared to 0.015419 seconds required to evaluate the function
Zygote.jacobian(
    (n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array) -> Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, kind, rotationally_symmetric, symmetric_about_plan_perpendicular_z),
     n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array
)


function T_matrix_SeparateRealImag(
    n_max::Int, k1_r::Real, k1_i::Real, k2_r::Real, k2_i::Real,
    r_array::AbstractArray, θ_array::AbstractArray, ϕ_array::AbstractArray, n̂_array::AbstractArray,
    rotationally_symmetric, symmetric_about_plan_perpendicular_z
)
    RgQ = Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, "regular", rotationally_symmetric, symmetric_about_plan_perpendicular_z)
    Q   = Q_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, "irregular", rotationally_symmetric, symmetric_about_plan_perpendicular_z)
    
    Q = Float64.(Q) # TODO: I need to make the concrete type programmable, in case I need Float128
    RgQ = Float64.(RgQ) # TODO: I need to make the concrete type programmable, in case I need Float128

    RgQ_r = RgQ[:, 1:Int(end / 2)]
    RgQ_i = RgQ[:, Int(end / 2) + 1:end]
    Q_r = Q[:, 1:Int(end / 2)]
    Q_i = Q[:, Int(end / 2) + 1:end]   

    invQ = complex_matrix_inversion(Q_r, Q_i)
    T = -1 .* complex_matrix_multiplication(RgQ_r, RgQ_i, invQ[1], invQ[2])
    return hcat(T[1], T[2])    
end


@time T = T_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
@time T_ = Tmatrix.T_matrix(n_max, complex(k1_r, k1_i), complex(k2_r, k2_i), complex(k1_r, k1_i) .* r_array, complex(k2_r, k2_i) .* r_array, r_array, θ_array, ϕ_array, n̂_array; rotationally_symmetric=rotationally_symmetric, symmetric_about_plan_perpendicular_z=false)

println("==================================================================")
println("Displaying results from _SeparateRealImag and the complex function")
println("Displaying result from _SeparateRealImag function --------------------")
display(get_complex_matrix_from_concatenated_real_imag(T))
println("Displaying result from complex function ------------------------------")
display(T_)

Zygote.jacobian(T_matrix_SeparateRealImag, n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plan_perpendicular_z)
