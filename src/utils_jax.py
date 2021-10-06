import jax.numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# Contains all helper defs for Tmatrix module



"""
Create a single index from m and n
We fill the index like this:
`
idx = 0
for n = 1:n_max    
    for m = -n:n
        global idx
        idx += 1
    

`
"""
def single_index_from_m_n(m:int, n:int):
    return n * (n + 1) + m


"""
    Get the maximum single index, given the maximum n.
"""
def get_max_single_index_from_n_max(n_max:int):
    return single_index_from_m_n(n_max, n_max)


"""
    Multiply the integrand by the `dS` element, which equals r²sin(θ)
"""
def surface_integrand(
        integrand, r_array, θ_array #TODO make types work
    ): #where {R <: Real,C <: Complex{R}} 
    return integrand * r_array**2 * np.sin(θ_array)


#TODO make sure this is just python np.trapz with x and y flipped
# def trapz_ELZOUKA(x::AbstractVector{R}, y::AbstractVector{N}): #where {R <: Real,N <: Number}
#     # TODO: small error if compared with Trapz.trapz
#     base = x[2:end] - x[1:end - 1]
#     av_height = (y[2:end] + y[1:end - 1]) / 2
#     areas = base .* av_height
#     total_area = sum(areas)
#     return total_area


# """
#     2D numerical integral using trapezoidal rule
# x and y are 1D arrays, z is 2D array
# """
# def trapz_ELZOUKA(x::AbstractVector{R}, y::AbstractVector{R}, z::AbstractMatrix{N}): #where {R <: Real,N <: Number}
#     integrand_wrt_x = trapz_ELZOUKA.(eachcol(repeat(x, 1, size(z, 2))), eachcol(z))
#     return trapz_ELZOUKA(y, integrand_wrt_x)

