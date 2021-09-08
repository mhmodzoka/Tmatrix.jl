# Contains all helper functions for Tmatrix module

using LinearAlgebra

"""
    Calculate dot product for two vectors.

# I made this function because the "dot" function in LinearAlgebra package doesn't work as expected for complex vectors.

Parameters

A, B : two vectors

# Returns

scalar dot product
"""
function vector_dot_product(A, B)
    return sum(A .* B)
end

"""
    Create a meshgrid, to be used with numerical integrals.

The meshgrid created is following this rule:
x_grid[dim1,dim2]
dim1 corresponds to x index
dim2 corresponds to y index
"""
function meshgrid(x, y)
    x_grid = repeat(x, 1, length(y))
    y_grid = repeat(y', length(x), 1)
    return x_grid, y_grid
end

"""
    Create meshgrid of θ and ϕ
"""
function meshgrid_θ_ϕ(
    n_θ_points,
    n_ϕ_points;
    min_θ = 1e-16,
    min_ϕ = 1e-16,
    rotationally_symmetric = false,
)
    #TODO can avoid min values w full algebra
    θ_1D_array = LinRange(min_θ, π, n_θ_points)
    ϕ_1D_array = LinRange(min_ϕ, 2π, n_ϕ_points)
    if rotationally_symmetric
        θ_array = collect(θ_1D_array)
        ϕ_array = zeros(size(θ_array))
    else
        θ_array, ϕ_array = meshgrid(θ_1D_array, ϕ_1D_array)
    end
    return θ_array, ϕ_array
end

"""
Create a single index from m and n
We fill the index like this:
`idx = 0 for n = 1:n_max     for m = -n:n global idx idx += 1 end end`
"""
function single_index_from_m_n(m::Int, n::Int)
    return n * (n + 1) + m
end

"""
    Get the maximum single index, given the maximum n.
"""
function get_max_single_index_from_n_max(n_max::Int)
    return single_index_from_m_n(n_max, n_max)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
    integrand::AbstractVecOrMat{C},
    r_array::AbstractVecOrMat{R},
    θ_array::AbstractVecOrMat{R},
) where {R <: Real, C <: Complex{R}}
    return integrand .* r_array .^ 2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
    integrand::AbstractVecOrMat{R},
    r_array::AbstractVecOrMat{R},
    θ_array::AbstractVecOrMat{R},
) where {R <: Real}
    return integrand .* r_array .^ 2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
    integrand::AbstractVecOrMat{Matrix{C}},
    r_array::AbstractVecOrMat{R},
    θ_array::AbstractVecOrMat{R},
) where {R <: Real, C <: Complex{R}}
    return integrand .* r_array .^ 2 .* sin.(θ_array)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(
    integrand::AbstractVecOrMat{Matrix{R}},
    r_array::AbstractVecOrMat{R},
    θ_array::AbstractVecOrMat{R},
) where {R <: Real, C <: Complex{R}}
    return integrand .* r_array .^ 2 .* sin.(θ_array)
end

"""
    Save T-matrix to HDF5 file, with fields "Tmatrix_real_CELES_convention" and "Tmatrix_imag_CELES_convention"
"""
function save_Tmatrix_to_HDF5_file(T, HDF5_filename)
    h5write(HDF5_filename, "Tmatrix_real_CELES_convention", Float64.(real(T))) # TODO: can we store BigFloat in HDF5? I converted to `Float64` because I couldn't read the file when I stored BigFloat
    h5write(HDF5_filename, "Tmatrix_imag_CELES_convention", Float64.(imag(T)))
end

# getting indices of T-matrix
"""
    Get 2D array, first and second columns are n and m values, respectively.

m = -n : 1 : +n
n will be the repeated value of input
"""
function get_n_m_array_given_n(n)
    return hcat(repeat([n], 2 * n + 1), (-n):n)
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

## Arguments

    n_max : Int, determine the size of the T-matrix

## Returns

    m_matrix, n_matrix, m__matrix, n__matrix : each is a square matrix
"""
function get_m_n_m__n__matrices_for_T_matrix(n_max)
    n_m_idx = get_n_m_idx_array_given_n_max(n_max)
    n_array = n_m_idx[:, 1]
    m_array = n_m_idx[:, 2]
    idx_array = n_m_idx[:, 3]

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
    idx = 0
    for n in 1:n_max
        for m in (-n):n
            idx += 1
            idx_ = 0
            for n_ in 1:n_max
                for m_ in (-n_):n_
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
    m_matrix, n_matrix, m__matrix, n__matrix =
        get_m_n_m__n__matrices_for_T_matrix_with_forloop(n_max)
    _m_matrix, _n_matrix, _m__matrix, _n__matrix =
        get_m_n_m__n__matrices_for_T_matrix(n_max)

    println(_n_matrix == n_matrix)
    println(_m_matrix == m_matrix)
    println(_n__matrix == n__matrix)
    println(_m__matrix == m__matrix)
end

#######################
# TOBE moved to a new package for trapezoidal integraion, that will be compatible with autodiff
"""
    1D numerical integral using trapezoidal rule
x and y are 1D arrays
"""
#TODO rename to not elzouka lol
function trapz_ELZOUKA(
    x::AbstractVector{R},
    y::AbstractVector{N},
) where {R <: Real, N <: Number}
    # TODO: small error if compared with Trapz.trapz
    base = x[2:end] - x[1:(end - 1)]
    av_height = (y[2:end] + y[1:(end - 1)]) / 2
    areas = base .* av_height
    total_area = sum(areas)
    return total_area
end

"""
    2D numerical integral using trapezoidal rule

x and y are 1D arrays, z is 2D array
"""
function trapz_ELZOUKA(
    x::AbstractVector{R},
    y::AbstractVector{R},
    z::AbstractMatrix{N},
) where {R <: Real, N <: Number}
    integrand_wrt_x = trapz_ELZOUKA.(eachcol(repeat(x, 1, size(z, 2))), eachcol(z))
    return trapz_ELZOUKA(y, integrand_wrt_x)
end

#######################
# Calculate orientation-averaged sscattering, extinction and absorption cross sections.
"""
    calculate orientation averaged scattering cross section given a T-matrix and a wavevector.

Input T-matrix can be a complex square matrix or a concatenation of separate real and imag parts of T-matrix
"""
function get_OrentationAv_scattering_CrossSection_from_Tmatrix(
    T::AbstractMatrix,
    k1::C,
) where {C <: Complex}
    if size(T)[1] == size(T)[2] # if T-matrix is a square matrix, then T-matrix is complex
        return real(2 * pi / k1^2 * sum(T .* conj(T)))
    elseif size(T)[1] == size(T)[2] / 2 # if T-matrix has number of columns double the number of rows, then T-matrix is hcat() of real and imag parts of Tmatrix
        return get_OrentationAv_scattering_CrossSection_from_Tmatrix_SeparateRealImag(
            T,
            real(k1),
            imag(k1),
        )
    end
end

"""
    calculate orientation averaged scattering cross section given a T-matrix, wavelength and optical properties of surrounding.

Input T-matrix can be a complex square matrix or a concatenation of separate real and imag parts of T-matrix
"""
function get_OrentationAv_scattering_CrossSection_from_Tmatrix(
    T::AbstractMatrix,
    wl_or_freq_input::R,
    input_unit::String,
    Eps_r_1::C,
    Mu_r_1::C,
) where {C <: Complex, R <: Real}
    return get_OrentationAv_scattering_CrossSection_from_Tmatrix(
        T,
        get_WaveVector(
            wl_or_freq_input;
            input_unit = input_unit,
            Eps_r = Eps_r_1,
            Mu_r = Mu_r_1,
        ),
    )
end

"""
    calculate orientation averaged scattering cross section given a T-matrix and a wavevector. This works for "_SeparateRealImag"
"""
function get_OrentationAv_scattering_CrossSection_from_Tmatrix_SeparateRealImag(
    T::AbstractMatrix{R},
    k1_r::R,
    k1_i::R,
) where {R <: Real}
    T_r, T_i = Tmatrix.separate_real_imag(T)
    T_by_conjT_sum = sum(complex_multiply.(T_r, T_i, T_r, -T_i))
    k1_squared = complex_multiply(k1_r, k1_i, k1_r, k1_i)
    two_pi_over_k1_squared = complex_divide(2 * pi, 0, k1_squared[1], k1_squared[2])
    return complex_multiply(
        two_pi_over_k1_squared[1],
        two_pi_over_k1_squared[2],
        T_by_conjT_sum[1],
        T_by_conjT_sum[2],
    )[1]
end

"""
    calculate orientation averaged extinction cross section given a T-matrix and a wavevector.
"""
function get_OrentationAv_extinction_CrossSection_from_Tmatrix(
    T::AbstractMatrix,
    k1::C,
) where {C <: Complex}
    if size(T)[1] == size(T)[2] # if T-matrix is a square matrix, then T-matrix is complex
        return real(-2 * pi / k1^2 * tr(real(T)))
    elseif size(T)[1] == size(T)[2] / 2 # if T-matrix has number of columns double the number of rows, then T-matrix is hcat() of real and imag parts of Tmatrix
        return get_OrentationAv_extinction_CrossSection_from_Tmatrix_SeparateRealImag(
            T,
            real(k1),
            imag(k1),
        )
    end
end

"""
    calculate orientation averaged extinction cross section given a T-matrix and a wavevector. This works for "_SeparateRealImag"
"""
function get_OrentationAv_extinction_CrossSection_from_Tmatrix_SeparateRealImag(
    T::AbstractMatrix{R},
    k1_r::R,
    k1_i::R,
) where {R <: Real}
    T_r, T_i = Tmatrix.separate_real_imag(T)
    k1_squared = complex_multiply(k1_r, k1_i, k1_r, k1_i)
    two_pi_over_k1_squared = complex_divide(2 * pi, 0, k1_squared[1], k1_squared[2])
    return -two_pi_over_k1_squared[1] * tr(T_r)
end

"""
    calculate orientation averaged absorption cross section given a T-matrix and a wavevector.
"""
function get_OrentationAv_absorption_CrossSection_from_Tmatrix(
    T::AbstractMatrix,
    k1::C,
) where {C <: Complex}
    return get_OrentationAv_extinction_CrossSection_from_Tmatrix(T, k1) -
           get_OrentationAv_scattering_CrossSection_from_Tmatrix(T, k1)
end

"""
    calculate orientation averaged absorption cross section given a T-matrix and a wavevector. This works for "_SeparateRealImag"
"""
function get_OrentationAv_absorption_CrossSection_from_Tmatrix_SeparateRealImag(
    T::AbstractMatrix{R},
    k1_r::R,
    k1_i::R,
) where {R <: Real}
    return get_OrentationAv_extinction_CrossSection_from_Tmatrix_SeparateRealImag(
        T,
        k1_r,
        k1_i,
    ) - get_OrentationAv_scattering_CrossSection_from_Tmatrix_SeparateRealImag(
        T,
        k1_r,
        k1_i,
    )
end

"""
    calculate orientation averaged emissivity cross section given a T-matrix and a wavevector.

`particle_surface_area` has to be in units consistent with wavevector units. If waveevctor unit is per m, then the particle surface area has to be in m^2
"""
function get_OrentationAv_emissivity_from_Tmatrix(
    T::AbstractMatrix,
    k1::C,
    particle_surface_area::R,
) where {R <: Real, C <: Complex}
    return get_OrentationAv_absorption_CrossSection_from_Tmatrix(T, k1) * 4 /
           particle_surface_area
end

"""
    calculate orientation averaged emissivity cross section given a T-matrix and a wavevector. This works for "_SeparateRealImag"

`particle_surface_area` has to be in units consistent with wavevector units. If waveevctor unit is per m, then the particle surface area has to be in m^2
"""
function get_OrentationAv_emissivity_from_Tmatrix_SeparateRealImag(
    T::AbstractMatrix{R},
    k1_r::R,
    k1_i::R,
    particle_surface_area::R,
) where {R <: Real}
    return get_OrentationAv_absorption_CrossSection_from_Tmatrix_SeparateRealImag(
        T,
        k1_r,
        k1_i,
    ) * 4 / particle_surface_area
end

"""
    calculate orientation averaged emissivity cross section given a T-matrix, wavelength and optical properties of surrounding. This works for "_SeparateRealImag"

`particle_surface_area` has to be in units consistent with wavevector units. If waveevctor unit is per m, then the particle surface area has to be in m^2
"""
function get_OrentationAv_emissivity_from_Tmatrix_SeparateRealImag(
    T::AbstractMatrix{R},
    wl_or_freq_input::R,
    input_unit::String,
    Eps_r_r_1::R,
    Eps_r_i_1::R,
    Mu_r_r_1::R,
    Mu_r_i_1::R,
    particle_surface_area::R,
) where {R <: Real}
    k1_complex = get_WaveVector(
        wl_or_freq_input;
        input_unit = input_unit,
        Eps_r = Complex(Eps_r_r_1, Eps_r_i_1),
        Mu_r = Complex(Mu_r_r_1, Mu_r_i_1),
    )
    return get_OrentationAv_absorption_CrossSection_from_Tmatrix_SeparateRealImag(
        T,
        k1_r,
        k1_i,
    ) * 4 / particle_surface_area
end
