# Calculation of T-matrix using only real numbers.
# Unlike the complex version, this is compatible with automatic differentiation

# TODO:
# 1- Find a way to enforce concrrete types at the T_matrix function. So far, I have defined abstract types as inputs for all functions. I am not sure how to convert to a concrete type mathcing my computation requirement (e.g., BigFloats)
# 2- Check why trapz_ELZOUKA is not returning exactly the same result as Trapz.trapz
# 3- The functions I have wrote separating real and imaginary parts are slower, figure out why
# 4- I need to get rid of Abstract Types, according to https://docs.julialang.org/en/v1/manual/performance-tips/ . For now, I am not sure how to make my code work with arbitrary float size
# -- I have tried to use r_array::Array{Float64}, but got an error when r_array is ::Vector{Float64}
# 5- Use StaticArrays when possible.

# NOTE: Lessons learned:
# 1- In order to be compatible with auto differentiation, all functions should return only one Array (no Tuples), which contains no nested arrays. This can be done by vcat(array...)

include("utils.jl")
include("geometry.jl")

using Tmatrix
using ComplexOperations
using LinearAlgebra
using VectorSphericalWaves
using Trapz

export calculate_Tmatrix_for_spheroid_SeparateRealImag

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
    cross_product_MN_dot_n̂ = (-1).^m .* complex_vector_dot_product.(cross_product_MN, n̂_array)

    J_integrand = Tmatrix.surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    return vcat(J_integrand...) # I had to flatten all nested arrays.
end    

function J_mn_m_n__SeparateRealImag(
        m::Int, n::Int, m_::Int, n_::Int,
        k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}, ϕ_array::AbstractVecOrMat{R}, n̂_array::Any, # TODO: I don't know why I get an error when I use n̂_array::AbstractVecOrMat{Vector{Float64}}
        kind::String, J_superscript::Int, rotationally_symmetric::Bool,
    ) where {R <: Real}

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


function Q_mn_m_n_SeparateRealImag(
        m::Int, n::Int, m_::Int, n_::Int,
        k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}, ϕ_array::AbstractVecOrMat{R}, n̂_array::Any, # TODO: how to define a multidimensional array type with Float64 elements?
        kind::String, Q_superscript::Int, rotationally_symmetric::Bool, symmetric_about_plan_perpendicular_z::Bool,    
    )  where {R <: Real}

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


function Q_matrix_SeparateRealImag(
        n_max::Int, k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}, ϕ_array::AbstractVecOrMat{R}, n̂_array::Any,
        kind::String, rotationally_symmetric::Bool, symmetric_about_plan_perpendicular_z::Bool,
    ) where {R <: Real}
    
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

function T_matrix_SeparateRealImag(
        n_max::Int, k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}, ϕ_array::AbstractVecOrMat{R}, n̂_array::Any,
        rotationally_symmetric, symmetric_about_plan_perpendicular_z
    ) where {R <: Real}

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

function calculate_Tmatrix_for_spheroid_SeparateRealImag(
        rx::R, rz::R, n_max::Int,
        k1_r::R, k1_i::R, k2_r::R, k2_i::R,
        n_θ_points=10, n_ϕ_points=20, HDF5_filename=nothing,
        rotationally_symmetric=false, symmetric_about_plan_perpendicular_z=false,
    ) where {R <: Real}
    
    θ_1D_array = LinRange(1e-16, π, n_θ_points);
    ϕ_1D_array = LinRange(1e-16, 2π, n_ϕ_points);
    if rotationally_symmetric        
        θ_array = collect(θ_1D_array)
        ϕ_array = zeros(size(θ_array))
    else
        # eventually, this should be removed. I just keep it for sanity checks.
        θ_array, ϕ_array = meshgrid(θ_1D_array, ϕ_1D_array);
    end
    θ_array = convert.(typeof(rx), θ_array)
    ϕ_array = convert.(typeof(rx), ϕ_array)
    r_array, n̂_array = ellipsoid(rx, rz, θ_array);    
    T = T_matrix_SeparateRealImag(n_max, k1_r, k1_i, k2_r, k2_i, r_array, θ_array, ϕ_array, n̂_array, rotationally_symmetric, symmetric_about_plan_perpendicular_z)    
    return T
end