# calculate T-matrix of individual particles using Null-Field method, using equations from:
# Mishchenko, M.I., Travis, L.D., and Lacis, A.A. (2002). Scattering, absorption, and emission of light by small particles (Cambridge University Press).

module Tmatrix
    

#############################################################
# import
#include("VectorSphericalHarmonicsVectorized.jl")
#import .VectorSphericalHarmonics
using VectorSphericalWaves
using StaticArrays
using EllipsisNotation
using LinearAlgebra
using Trapz
using HDF5

#using HCubature # I may consider this package for numerical integrals

export calculate_Tmatrix_for_spheroid
#############################################################
# Inputs
# - surrounding material (e.g., dielectric constant, refractive index, wavevector)
# - particle material (e.g., dielectric constant, refractive index, wavevector)
# - angular_resolution
# - n_max
# - particle geometry: it can be:
# -- a function r(θ) (i.e., axi-symmetric particle)
# -- a function r(θ,ϕ) (i.e., arbitrary 3D particle)
# -- a meshfile
# -- a string defining geometry class (e.g., "cylinder", "spheroid", "ellipsoid", etc.)

# output
# - T_matrix,

#############################################################
function dot_ELZOUKA(A, B)
    return sum(A .* B)
end

function cross_ELZOUKA(A, B)
    return [
        A[2]*B[3] - A[3]*B[2],
        A[3]*B[1] - A[1]*B[3],
        A[1]*B[2] - A[2]*B[1],
    ]
end

""" calculate spherical coordinates arrays

 - this can be either 1D arrays for axi-symmetric particles or 2D arrays for general 3D particles
 """
function get_r_θ_ϕ_arrays(geometry; angular_resolution = 0.5)
    # TODO: use a package that takes a mesh and do integrals
    # check: https://github.com/jareeger/Smooth_Closed_Surface_Quadrature_RBF-julia

    if geometry isa String
        if geometry in ["wire", "cylinder"]
        elseif geometry in ["spheroid", "ellipsoid"]
        end
    elseif geometry isa Function
        # TODO: check if the function has one or two arguments

    else
        throw(DomainError("Please define geometry as 'String', 'Function', or 'Filepath'"))
    end
end

"""
Create a meshgrid, to be used with numerical integrals.
The meshgrid created is following this rule:
x_grid[dim1,dim2]
dim1 corresponds to x index
dim2 corresponds to y index
"""
function meshgrid(x,y)    
    x_grid = repeat(x , 1, length(y))
    y_grid = repeat(y', length(x),1)
    return x_grid, y_grid
end

#############################################################
"calculate M_mn_wave and N_mn_wave as an array"
function M_mn_wave_array(m, n, kr_array, θ_array, ϕ_array; kind="regular", use_Alok_vector_preallocation=true)
    """
    This is the same as VectorSphericalHarmonics.M_mn_wave, but accept kr_array, θ_array, ϕ_array
    Parameters
    ==========
    kr_array, θ_array, ϕ_array : arrays of arbitrary shape

    return
    ======
    M_mn_wave_array_ : M_mn_wave with shape same as any of kr_array, θ_array, ϕ_array, with an added dimension to represent the three components
    """

    if use_Alok_vector_preallocation
        # Alok way is faster indeed!
        # TODO: @Alok, I think if we use boradcast it would be faster. I think avoiding preallocation makes the code cleaner and faster
        M_mn_wave_array_ = (_-> zero(SVector{3,Complex})).(kr_array)
        for idx in eachindex(kr_array)
            M_mn_wave_array_[idx] = M_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind = kind)
        end

    else
        # EllipsisNotation way
        M_mn_wave_array_ = zeros(Complex, append!(collect(size(kr_array)), 3)...)
        #M_mn_wave_array_ = zeros(Complex, append!(collect(size(kr_array)), 3))
        for idx in CartesianIndices(kr_array)
            M_mn_wave_array_[idx,:] = M_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind = kind)
        end
    end

    return M_mn_wave_array_

end


function N_mn_wave_array(m, n, kr_array, θ_array, ϕ_array; kind="regular", use_Alok_vector_preallocation=true)
    """
    This is the same as VectorSphericalHarmonics.N_mn_wave, but accept kr_array, θ_array, ϕ_array
    Parameters
    ==========
    kr_array, θ_array, ϕ_array : arrays of arbitrary shape

    return
    ======
    N_mn_wave_array_ : N_mn_wave with shape same as any of kr_array, θ_array, ϕ_array, with an added dimension to represent the three components
    """

    if use_Alok_vector_preallocation
        # Alok way
        N_mn_wave_array_ = (_-> zero(SVector{3,Complex})).(kr_array)
        for idx in eachindex(kr_array)
            N_mn_wave_array_[idx] = N_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind = kind)
        end

    else
        # EllipsisNotation way
        N_mn_wave_array_ = zeros(Complex, append!(collect(size(kr_array)), 3)...)
        #N_mn_wave_array_ = zeros(Complex, append!(collect(size(kr_array)), 3))
        for idx in CartesianIndices(kr_array)
            N_mn_wave_array_[idx,:] = N_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind = kind)
        end
    end

    return N_mn_wave_array_

end

#############################################################
# calculate J and Rg J, from equations 5.184 and 5.190
function J_mn_m_n__integrand(
    m,
    n,
    m_,
    n_,
    k1r_array,
    k2r_array,
    r_array,
    θ_array,
    ϕ_array,
    n̂_array;
    kind = "regular",
    J_superscript = 11,
    use_Alok_vector_preallocation = true
)
    """
    Calculate the integrand of J (including dS=r²sin(θ)dθdϕ) from equations 5.184 and 5.190.
    It calculate all the integrand, and multiply it by r²sin(θ)
    
    Inputs
    ======
    m, n, m_, n_ : int, order and rank of VSWF inside and outside the particle
    k1r_array, k2r_array : complex 2D array, product of wavevector and r, evaluated at points on the surface of the particle
    θ_array, ϕ_array : complex 2D array, spherical coordinates of points on the particle surface
    n̂_array : 3D array, unit vector normal to the surface of the particle
    kind : string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
    J_superscript : superscript at the top of J, it can be any of [11,12,21,22]

    Outputs
    =======
    J
    """

    # calculate the integrand
    if J_superscript == 11 # TODO: this if-statement can be done more nicely. We separate J_superscript into two pieces, the number 1 represents M_mn_wave_array, while number 2 represents N_mn_wave_array        
        first_function = M_mn_wave_array
        second_function = M_mn_wave_array
    elseif J_superscript == 12
        first_function = M_mn_wave_array
        second_function = N_mn_wave_array
    elseif J_superscript == 21
        first_function = N_mn_wave_array
        second_function = M_mn_wave_array
    elseif J_superscript == 22
        first_function = N_mn_wave_array
        second_function = N_mn_wave_array
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

    if use_Alok_vector_preallocation
        # the cross product    
        cross_product_MN = cross.(
            first_function( m_, n_, k2r_array, θ_array, ϕ_array, kind = kind_first_function, use_Alok_vector_preallocation=use_Alok_vector_preallocation),
            second_function(-m, n , k1r_array, θ_array, ϕ_array, kind = kind_second_function, use_Alok_vector_preallocation=use_Alok_vector_preallocation)
        )

        # dot multiplying the cross product by the unit vector, and multiplying by (-1)^m
        cross_product_MN_dot_n̂ = (-1).^m .* dot_ELZOUKA.(cross_product_MN, n̂_array)
    else
        # the cross product  
        cross_product_MN = cross_product_on_last_dimension(
            first_function( m_, n_, k2r_array, θ_array, ϕ_array, kind = kind_first_function, use_Alok_vector_preallocation=use_Alok_vector_preallocation),
            second_function(-m, n , k1r_array, θ_array, ϕ_array, kind = kind_second_function, use_Alok_vector_preallocation=use_Alok_vector_preallocation)
        )
        # dot multiplying the cross product by the unit vector, and multiplying by (-1)^m
        cross_product_MN_dot_n̂ = zeros(Complex, size(cross_product_MN)[1:end-1])
        '''
        for idx in CartesianIndices(size(cross_product_MN)[1:end-1])        
            cross_product_MN_dot_n̂[idx] = (-1)^m * dot_ELZOUKA(n̂_array[idx,:], cross_product_MN[idx,:])
        end
    end
    
    J_integrand = surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    return J_integrand
end

function surface_integrand(integrand, r_array, θ_array)
    return integrand .* r_array.^2 .* sin.(θ_array)
end

#############################################################
#angular_resolution = 0.05
#θ_array = collect(1e-16:angular_resolution:π)
function ellipsoid(rx, rz, θ_array; use_Alok_vector_preallocation=true)
    """
    returns r and n̂ coordinate as a function of θ_array
    """
    r = rx .* rz ./ sqrt.((rx .* cos.(θ_array)) .^ 2 + (rz .* sin.(θ_array)) .^ 2)
    ∂r_by_∂θ = (rx .^ 2 - rz .^ 2) / (rx .^ 2 * rz .^ 2) .* r .^ 3 .* sin.(θ_array) .* cos.(θ_array)

    n̂_r_comp = r ./ sqrt.(r .^ 2 + ∂r_by_∂θ .^ 2)
    n̂_θ_comp = -∂r_by_∂θ ./ sqrt.(r .^ 2 + ∂r_by_∂θ .^ 2)
    n̂_ϕ_comp = zeros(Real, size(r))

    if use_Alok_vector_preallocation
        # TODO @Alok, is there an efficient way to do it?
        # this one is maybe 10x slower than the one in "else"
        n̂ = (_-> zero(SVector{3,Real})).(r)
        for idx in CartesianIndices(n̂)
            n̂[idx] = [
                n̂_r_comp[idx],
                n̂_θ_comp[idx],
                n̂_ϕ_comp[idx],
            ]
        end
    else        
        n̂ = hcat(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)
    end 
    
    return r, n̂
end

#############################################################
# ..., 3
function cross_product_on_last_dimension(A, B)
    """
    Calculate cross product of two vectors A,B, assuming that the three components are encoded at the last index
    """
    A_size = size(A)
    B_size = size(B)

    if (A_size[end] ≠ 3 || B_size[end] ≠ 3) || (A_size ≠ B_size)
        throw(DomainError("both arrays must have the third dimension size as 3"))
    end

    product_result = zeros(A_size)

    A_1 = A[..,1]
    A_2 = A[..,2]
    A_3 = A[..,3]

    B_1 = B[..,1]
    B_2 = B[..,2]
    B_3 = B[..,3]

    for idx in CartesianIndices(A_1)        
        product_result[idx,:] = cross([A_1[idx], A_2[idx], A_3[idx]], [B_1[idx], B_2[idx], B_3[idx]])
    end
    
    """
    if length(A_size) == 1
        product_result = cross(A, B)

    elseif length(A_size) == 2
        for idx1 in range(1, stop = A_size[1])
            product_result[idx1, :] = cross(A[idx1, :], B[idx1, :])
        end
    elseif length(A_size) == 3
        for idx1 in range(1, stop = A_size[1])
            for idx2 in range(1, stop = A_size[2])
                product_result[idx1, idx2, :] = cross(A[idx1, idx2, :], B[idx1, idx2, :])
            end
        end
    end
    """

    return product_result
end


# TODO: @Alok, interal will be evaluated here
function J_mn_m_n_(
    m,
    n,
    m_,
    n_,
    k1r_array,
    k2r_array,
    r_array,
    θ_array,
    ϕ_array,
    n̂_array;
    kind = "regular",
    J_superscript = 11,
    use_Alok_vector_preallocation = true
)
    
    J_integrand_dS = J_mn_m_n__integrand(
        m,
        n,
        m_,
        n_,
        k1r_array,
        k2r_array,
        r_array,
        θ_array,
        ϕ_array,
        n̂_array;
        kind = kind,
        J_superscript = J_superscript,
        use_Alok_vector_preallocation = use_Alok_vector_preallocation
    )

    
    # TODO: @Alok
    # J_integrand_dS is everything inside the integral. I just need to integrate this with respect to θ [0,π] and ϕ [0,2π]. Here, I will exploit particle symmetry to simplify the integral [later].
    # I was thinking that using HCubature is not optimum, as it may take very long time. It took a long time to evaluate ∫sin,cos.
    # I am thinking of using this one: https://github.com/JuliaApproximation/FastGaussQuadrature.jl
    # or a simpler trapz integral: https://github.com/francescoalemanno/Trapz.jl to make sure my matrices have the same size in all cases, which can result in a faster code, and we can afford using fine discretization of θ and ϕ

    # integrate over θ and ϕ
    # assuming that θ_array, ϕ_array were created with meshgrid function
    J = trapz((θ_array[:,1], ϕ_array[1,:]), J_integrand_dS)

    return J
    
end


function Q_mn_m_n_(
    m,
    n,
    m_,
    n_,
    k1,
    k2,
    k1r_array,
    k2r_array,
    r_array,
    θ_array,
    ϕ_array,
    n̂_array;
    kind = "regular",
    Q_superscript = 11,
    use_Alok_vector_preallocation = true,    
)    
    if     Q_superscript == 11; J_superscript_1 = 21 ; J_superscript_2 = 12
    elseif Q_superscript == 12; J_superscript_1 = 11 ; J_superscript_2 = 22
    elseif Q_superscript == 21; J_superscript_1 = 22 ; J_superscript_2 = 11
    elseif Q_superscript == 22; J_superscript_1 = 12 ; J_superscript_2 = 21
    end    

    Q = (
        -im .* k1 .* k2 .* J_mn_m_n_(m,n,m_,n_,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array; kind = kind, J_superscript = J_superscript_1, use_Alok_vector_preallocation = use_Alok_vector_preallocation) 
        -im .* k1.^2    .* J_mn_m_n_(m,n,m_,n_,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array; kind = kind, J_superscript = J_superscript_2, use_Alok_vector_preallocation = use_Alok_vector_preallocation)
    )

    return Q
end

function Q_matrix(
    n_max,
    k1,
    k2,
    k1r_array,
    k2r_array,
    r_array,
    θ_array,
    ϕ_array,
    n̂_array;
    kind = "regular",    
    use_Alok_vector_preallocation = true,
    verbose = false
)
    idx_max = get_max_single_index_from_n_max(n_max)
    Q_mn_m_n_11 = zeros(Complex, idx_max,idx_max)
    Q_mn_m_n_12 = zeros(Complex, idx_max,idx_max)
    Q_mn_m_n_21 = zeros(Complex, idx_max,idx_max)
    Q_mn_m_n_22 = zeros(Complex, idx_max,idx_max)

    idx = 0;
    for n = 1:n_max
        for m = -n:n
            #global idx
            idx += 1            
            idx_ = 0;
            for n_ = 1:n_max
                for m_ = -n_:n_                    
                    idx_ += 1                    
                    if verbose; println("n,m,idx = $n,$m,$idx  n_,m_,idx_ = $n_,$m_,$idx_"); end
                    Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m,n,m_,n_,k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = kind,Q_superscript = 11,use_Alok_vector_preallocation = use_Alok_vector_preallocation)
                    Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m,n,m_,n_,k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = kind,Q_superscript = 12,use_Alok_vector_preallocation = use_Alok_vector_preallocation)
                    Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m,n,m_,n_,k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = kind,Q_superscript = 21,use_Alok_vector_preallocation = use_Alok_vector_preallocation)
                    Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m,n,m_,n_,k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = kind,Q_superscript = 22,use_Alok_vector_preallocation = use_Alok_vector_preallocation)                    
                end
            end
        end
    end
    Q = vcat(
        (hcat(Q_mn_m_n_11, Q_mn_m_n_12)),
        (hcat(Q_mn_m_n_21, Q_mn_m_n_22))
    )
    return Q
end

function T_matrix(
    n_max,
    k1,
    k2,
    k1r_array,
    k2r_array,
    r_array,
    θ_array,
    ϕ_array,
    n̂_array;    
    use_Alok_vector_preallocation = true,
    verbose = false,
    create_new_arrays = false,
    HDF5_filename = nothing
)   
    if create_new_arrays
        RgQ =       Q_matrix(n_max, k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = "regular"  ,use_Alok_vector_preallocation = true)
        Q_inv = inv(Q_matrix(n_max, k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = "irregular",use_Alok_vector_preallocation = true))
        T = -1 .* RgQ * Q_inv
    
    else
        T = (
            -     Q_matrix(n_max, k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = "regular"  ,use_Alok_vector_preallocation = true)
            * inv(Q_matrix(n_max, k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;kind = "irregular",use_Alok_vector_preallocation = true))
        )
    end

    if HDF5_filename != nothing
        save_Tmatrix_to_HDF5_file(T, HDF5_filename)
    end

    return T
end

"""
Create a single index from m and n
We fill the index like this:
idx = 0
for n = 1:n_max    
    for m = -n:n
        global idx
        idx += 1
    end
end
"""
function single_index_from_m_n(m,n)
    return n*(n + 1) + m
end

function get_max_single_index_from_n_max(n_max)
    return single_index_from_m_n(n_max,n_max)
end


function calculate_Tmatrix_for_spheroid(
        rx,rz,n_max,k1,k2;
        n_θ_points=10, n_ϕ_points=20, use_Alok_vector_preallocation = true, HDF5_filename = nothing)
    θ_1D_array = LinRange(1e-16, π, n_θ_points);
    ϕ_1D_array = LinRange(1e-16, 2π, n_ϕ_points);
    θ_array,ϕ_array = meshgrid(θ_1D_array,ϕ_1D_array);
    r_array,n̂_array = ellipsoid(rx, rz, θ_array; use_Alok_vector_preallocation=use_Alok_vector_preallocation);
    k1r_array = k1 .* r_array;
    k2r_array = k2 .* r_array;
    T = T_matrix(n_max,k1,k2,k1r_array,k2r_array,r_array,θ_array,ϕ_array,n̂_array;    use_Alok_vector_preallocation = use_Alok_vector_preallocation, HDF5_filename=HDF5_filename)    
    return T
end

function save_Tmatrix_to_HDF5_file(T, HDF5_filename)
    """
    Save T-matrix to HDF5 file, with fields "Tmatrix_real_CELES_convention" and "Tmatrix_imag_CELES_convention"
    """
    h5write(HDF5_filename, "Tmatrix_real_CELES_convention", real(T))
    h5write(HDF5_filename, "Tmatrix_imag_CELES_convention", imag(T))
end

end

"""
#############################################################
# testing
use_Alok_vector_preallocation = true
rx = 1; rz = 2;
n_max = 2
k1 = 1
k2 = 1.5 + 0.01*im
n_θ_points = 10
n_ϕ_points = 20

@time T = calculate_Tmatrix_for_spheroid(rx,rz,n_max,k1,k2; n_θ_points=n_θ_points, n_ϕ_points=n_ϕ_points, use_Alok_vector_preallocation = use_Alok_vector_preallocation)

println("Done with my 1st T-matrix!")









########################
# comparing [ for x,y,z] with use_Alok_vector_preallocation
# CONCLUSION: use_Alok_vector_preallocation is faster by 25%
θ_array = LinRange(1e-16, π, 100);
ϕ_array = LinRange(1e-16, 2π, 100);
θ_grid,ϕ_grid = meshgrid(θ_array,ϕ_array);
m_ = 2; n_ = 3; k2r_array = ones(size(θ_grid));
@time M_mn_wave_array(m_, n_, k2r_array, θ_grid, ϕ_grid, kind = "regular", use_Alok_vector_preallocation=true);
# 14.038013 seconds (60.00 M allocations: 2.012 GiB, 62.23% gc time)

k2r_array = 1;
@time [VSWF.M_mn_wave(m_, n_, k2r_array, θ, ϕ, kind = "regular") for θ=θ_array,ϕ=ϕ_array];
# 17.193079 seconds (44.18 M allocations: 1.726 GiB, 69.66% gc time)

k2r_array = 1;
@time VSWF.M_mn_wave.(m_, n_, k2r_array, θ_grid, ϕ_grid, kind = "regular");
# 16.609237 seconds (34.44 M allocations: 1.632 GiB, 69.76% gc time)
"""