# Contains all helper functions for Tmatrix module
"""
    Calculate dot product for two vectors.    
I made this function because the "dot" function in LinearAlgebra package doesn't work as expected for complex vectors.
Parameters
==========
A, B : two vectors

Returns
=======
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
Create a single index from m and n
We fill the index like this:
`
idx = 0
for n = 1:n_max    
    for m = -n:n
        global idx
        idx += 1
    end
end
`
"""
function single_index_from_m_n(m, n)
    return n * (n + 1) + m
end

"""
    Get the maximum single index, given the maximum n.
"""
function get_max_single_index_from_n_max(n_max)
    return single_index_from_m_n(n_max, n_max)
end

"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
function surface_integrand(integrand, r_array, θ_array)
    return integrand .* r_array.^2 .* sin.(θ_array)
end

"""
    Save T-matrix to HDF5 file, with fields "Tmatrix_real_CELES_convention" and "Tmatrix_imag_CELES_convention"
"""
function save_Tmatrix_to_HDF5_file(T, HDF5_filename)    
    h5write(HDF5_filename, "Tmatrix_real_CELES_convention", real(T))
    h5write(HDF5_filename, "Tmatrix_imag_CELES_convention", imag(T))
end

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


