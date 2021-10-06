# calculation of T-matrix using complex numbers.
# the only drawback of this is that it is not compatible with automatic differentiation

# TODO:
# 1- Use StaticArrays when possible.

export calculate_Tmatrix_for_spheroid

include("utils.jl")
include("geometry.jl")

# import
using VectorSphericalWaves
using StaticArrays
using EllipsisNotation
using LinearAlgebra
using Trapz
using HDF5

"""
    Calculate M_mn_wave as an array
This is the same as VectorSphericalHarmonics.M_mn_wave, but accept kr_array, θ_array, ϕ_array
Parameters
==========
kr_array, θ_array, ϕ_array : arrays of arbitrary shape

return
======
M_mn_wave_array_ : M_mn_wave with shape same as any of kr_array, θ_array, ϕ_array, with an added dimension to represent the three components
"""
def M_mn_wave_array(
        m: int, n: int, kr_array: AbstractVecOrMat{<:Complex{<:Real}}, θ_array: AbstractVecOrMat{R},
        ϕ_array: AbstractVecOrMat{R}; kind="regular"
    ): #where R <: Real
    # Alok way is faster indeed!
    # TODO: @Alok, I think if we use boradcast it would be faster. I think avoiding preallocation makes the code cleaner and faster
    M_mn_wave_array_ = (lambda _ : zero(SVector{3,Complex})).(kr_array)
    for idx in eachindex(kr_array)
        M_mn_wave_array_[idx] = M_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind=kind)
    
    return M_mn_wave_array_



"""
    Calculate N_mn_wave as an array
This is the same as VectorSphericalHarmonics.N_mn_wave, but accept kr_array, θ_array, ϕ_array
Parameters
==========
kr_array, θ_array, ϕ_array : arrays of arbitrary shape

return
======
N_mn_wave_array_ : N_mn_wave with shape same as any of kr_array, θ_array, ϕ_array, with an added dimension to represent the three components
"""
def N_mn_wave_array(
        m: int, n: int, kr_array: AbstractVecOrMat{<:Complex{<:Real}}, θ_array: AbstractVecOrMat{R},
        ϕ_array: AbstractVecOrMat{R}; kind="regular"
    ): #where {R <: Real}
    # Alok way
    N_mn_wave_array_ = (_ -> zero(SVector{3,Complex})).(kr_array)
    for idx in eachindex(kr_array)
        N_mn_wave_array_[idx] = N_mn_wave(m, n, kr_array[idx], θ_array[idx], ϕ_array[idx], kind=kind)
    
    return N_mn_wave_array_


#############################################################
# calculate J and Rg J, from equations 5.184 and 5.190
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
def J_mn_m_n__integrand(
        m: int, n: int, m_: int, n_: int,
        k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
        r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
        kind="regular", J_superscript=11
    ): #where {R <: Real, C <: Complex{R}}

    
    # determining the type of the first the second VSWF
    if J_superscript == 11 # TODO: this if-statement can be done more nicely. We separate J_superscript into two pieces, the number 1 represents M_mn_wave_array, while number 2 represents N_mn_wave_array
        first_function = M_mn_wave_array
        second_function = M_mn_wave_array
    elif J_superscript == 12
        first_function = M_mn_wave_array
        second_function = N_mn_wave_array
    elif J_superscript == 21
        first_function = N_mn_wave_array
        second_function = M_mn_wave_array
    elif J_superscript == 22
        first_function = N_mn_wave_array
        second_function = N_mn_wave_array
    else
        throw(DomainError("J_superscript must be any of [11,12,21,22]"))
    

    # determining the type of the first and second VSWF
    kind_first_function = "regular"
    if kind == "irregular"
        kind_second_function = "irregular"
    elif kind == "regular"
        kind_second_function = "regular"
    else
        throw(DomainError("""kind must be any of ["regular", "irregular"]"""))
    

    # the cross product
    cross_product_MN = cross.(
        first_function(m_, n_, k2r_array, θ_array, ϕ_array, kind=kind_first_function), # I can directly call M,N waves, with dots.
        second_function(-m, n, k1r_array, θ_array, ϕ_array, kind=kind_second_function)
    )

    # dot multiplying the cross product by the unit vector, and multiplying by (-1)^m
    cross_product_MN_dot_n̂ = (-1).^m .* vector_dot_product.(cross_product_MN, n̂_array)

    # multiplying by dS=r²sin(θ)
    J_integrand = surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    return J_integrand



"""
    Same as J_mn_m_n__integrand, but uses M_mn_wave_SVector and N_mn_wave_SVector
"""
def J_mn_m_n__integrand_SVector(
    m: int, n: int, m_: int, n_: int,
    k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
    r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
    kind="regular", J_superscript=11
): #where {R <: Real, C <: Complex{R}}


# determining the type of the first the second VSWF
if J_superscript == 11 # TODO: this if-statement can be done more nicely. We separate J_superscript into two pieces, the number 1 represents M_mn_wave_SVector, while number 2 represents N_mn_wave_SVector
    first_function = M_mn_wave_SVector
    second_function = M_mn_wave_SVector
elif J_superscript == 12
    first_function = M_mn_wave_SVector
    second_function = N_mn_wave_SVector
elif J_superscript == 21
    first_function = N_mn_wave_SVector
    second_function = M_mn_wave_SVector
elif J_superscript == 22
    first_function = N_mn_wave_SVector
    second_function = N_mn_wave_SVector
else
    throw(DomainError("J_superscript must be any of [11,12,21,22]"))


# determining the type of the first and second VSWF
kind_first_function = "regular"
if kind == "irregular"
    kind_second_function = "irregular"
elif kind == "regular"
    kind_second_function = "regular"
else
    throw(DomainError("""kind must be any of ["regular", "irregular"]"""))


# the cross product
cross_product_MN = cross.(
    first_function.(m_, n_, k2r_array, θ_array, ϕ_array, kind=kind_first_function), # I can directly call M,N waves, with dots.
    second_function.(-m, n, k1r_array, θ_array, ϕ_array, kind=kind_second_function)
)

# dot multiplying the cross product by the unit vector, and multiplying by (-1)^m
cross_product_MN_dot_n̂ = (-1).^m .* vector_dot_product.(cross_product_MN, n̂_array)

# multiplying by dS=r²sin(θ)
J_integrand = surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

return J_integrand


def J_mn_m_n_(
        m: int, n: int, m_: int, n_: int,
        k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
        r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
        kind="regular", J_superscript=11, rotationally_symmetric=False,
    ): #where {R <: Real, C <: Complex{R}}
    if rotationally_symmetric
        # make sure that θ_array is 1D
        if len(size(θ_array)) != 1
            throw(DomainError("Since you have indicated << rotationally_symmetric = true >>, θ_array has to be 1D. Now it is $(len(size(θ_array)))D"))
        
        ϕ_array = convert(type(θ_array), zeros(size(θ_array)))
    

    # getting the integrand
    if rotationally_symmetric && (m != m_)
        # the integral over ϕ is 2π * δ_m_m_, so it is zero if m != m_
        J_integrand_dS = zeros(size(θ_array))

    else
        J_integrand_dS = J_mn_m_n__integrand_SVector(
            m, n,m_,n_,
            k1r_array,k2r_array,
            r_array,θ_array,ϕ_array,n̂_array;
            kind=kind,J_superscript=J_superscript
        )
    

    # calculate the surface integral
    if rotationally_symmetric
        # integrate over θ only
        J = 2π * trapz((θ_array), J_integrand_dS)
    else
        # integrate over θ and ϕ
        # TODO: replace this integral with surface mesh quadrature, like this one: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5095443/ , https://github.com/jareeger/Smooth_Closed_Surface_Quadrature_RBF-julia
        # assuming that θ_array, ϕ_array were created with meshgrid function
        J = trapz((θ_array[:,1], ϕ_array[1,:]), J_integrand_dS)


    return J


def Q_mn_m_n_(
        m: int, n: int, m_: int, n_: int,
        k1: C, k2: C,
        k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
        r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
        kind="regular", Q_superscript=11, rotationally_symmetric=False,
    ): #where {R <: Real, C <: Complex{R}}
    if Q_superscript == 11; J_superscript_1 = 21 ; J_superscript_2 = 12
    elif Q_superscript == 12; J_superscript_1 = 11 ; J_superscript_2 = 22
    elif Q_superscript == 21; J_superscript_1 = 22 ; J_superscript_2 = 11
    elif Q_superscript == 22; J_superscript_1 = 12 ; J_superscript_2 = 21
    

    Q = (
        -im .* k1 .* k2 .* J_mn_m_n_(m, n, m_, n_, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript_1, rotationally_symmetric=rotationally_symmetric)
        - im .* k1.^2    .* J_mn_m_n_(m, n, m_, n_, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array; kind=kind, J_superscript=J_superscript_2, rotationally_symmetric=rotationally_symmetric)
    )

    return Q


def Q_matrix(
        n_max: int,
        k1: C, k2: C,
        k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
        r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
        kind="regular", rotationally_symmetric=False, symmetric_about_plane_perpicular_z=False,
        verbose=False,
    ): #where {R <: Real, C <: Complex{R}}
    idx_max = get_max_single_index_from_n_max(n_max)
    Q_mn_m_n_11 = zeros(type(k1), idx_max, idx_max) # TODO: @Alok, should I replace arrays with SMatrix? I am afraid it may get slower, as I have seen that StaticArray may get slower for arrays larger than 100 elements
    Q_mn_m_n_12 = zeros(type(k1), idx_max, idx_max)
    Q_mn_m_n_21 = zeros(type(k1), idx_max, idx_max)
    Q_mn_m_n_22 = zeros(type(k1), idx_max, idx_max)

    idx = 0;
    for n = range(1,n_max)
        for m = -n:n
            idx += 1
            idx_ = 0;
            for n_ = range(1,n_max)
                for m_ = -n_:n_
                    idx_ += 1
                    if verbose; println("n,m,idx = $n,$m,$idx  n_,m_,idx_ = $n_,$m_,$idx_"); 

                    if rotationally_symmetric
                        if m == m_
                            if symmetric_about_plane_perpicular_z
                                # apply equations 5.208 and 5.209
                                if iseven(n + n_)
                                    Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                    Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                                else
                                    Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                    Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                                
                            else
                                Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                            
                        
                    else
                        if symmetric_about_plane_perpicular_z && (m == m_)
                            # apply equations 5.208 and 5.209
                            if iseven(n + n_)
                                Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                            else
                                Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                            
                        else
                            Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                        
            
        
    
        
    
    Q = vcat(
        (hcat(Q_mn_m_n_11, Q_mn_m_n_12)),
        (hcat(Q_mn_m_n_21, Q_mn_m_n_22))
    )
    return Q


def T_matrix(
        n_max: int,
        k1: C, k2: C,
        k1r_array: AbstractVecOrMat{C}, k2r_array: AbstractVecOrMat{C},
        r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
        rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False, HDF5_filename=None,
        verbose=False, create_new_arrays=False, BigFloat_precision = None
    ): #where {R <: Real, C <: Complex{R}}

    if BigFloat_precision != None:
        def T_matrix_BigFloat(n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
                HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,  
                ):
            return T_matrix(
                n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
                HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,                
            
        return setprecision(BigFloat_precision) do
            return T_matrix(
                n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
                HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,                
            )            
        
    else
        if create_new_arrays
            RgQ =       Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="regular"  , rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)
            Q_inv = inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="irregular", rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))
            T = -1 .* RgQ * Q_inv

        else
            T = (
                -     Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="regular"  , rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)
                * inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="irregular", rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))
            )
        

        if HDF5_filename != None
            save_Tmatrix_to_HDF5_file(T, HDF5_filename)
        

        return T
    


"""
    Calculate T-matrix for a spheroid, with an extra kwarg `BigFloat_precision`

BigFloat_precision : set to `None` by default. When BigFloat_precision == None, then the function will use input types as they are.
"""
def calculate_Tmatrix_for_spheroid(
        rx: R, rz: R, n_max: int,
        k1: Complex{R}, k2: Complex{R};
        n_θ_points=10, n_ϕ_points=20, HDF5_filename=None,
        rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False,
        BigFloat_precision = None
    ): #where {R <: Real}    
    
    # create a grid of θ_ϕ
    θ_array, ϕ_array = meshgrid_θ_ϕ(n_θ_points, n_ϕ_points; min_θ=1e-16, min_ϕ=1e-16, rotationally_symmetric=rotationally_symmetric)    
    
    # calculate r and n̂ for the geometry
    r_array, n̂_array = ellipsoid(rx, rz, θ_array);

    # calculate T-matrix
    k1r_array = k1 .* r_array;
    k2r_array = k2 .* r_array;
    T = T_matrix(
        n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;
        HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,
        BigFloat_precision = BigFloat_precision
    )
    return T


"""
    General and nice wrapper for T-matric
Inputs
surrounding material (e.g., dielectric constant, refractive index, wavevector)
particle material (e.g., dielectric constant, refractive index, wavevector)
angular_resolution
n_max
particle geometry: it can be:
- function ``r(θ)`` (i.e., axi-symmetric particle)
- a function ``r(θ,ϕ)`` (i.e., arbitrary 3D particle)
- a meshfile
- a string defining geometry class (e.g., "cylinder", "spheroid", "ellipsoid", etc.)

# output
# - T_matrix,
"""
def Tmatrix_nice()
    # TODO



### precompiling
"""
R =
precompile(T_matrix, (
    Int, Complex{R}, k2: Complex{R},
    k1r_array: AbstractVecOrMat{<:Complex{<:Real}}, k2r_array: AbstractVecOrMat{<:Complex{<:Real}},
    r_array: AbstractVecOrMat{R}, θ_array: AbstractVecOrMat{R}, ϕ_array: AbstractVecOrMat{R}, n̂_array: Any; # TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
    rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False, HDF5_filename=None,
    verbose=False, create_new_arrays=False,)
"""
