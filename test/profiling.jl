# %%
using Tmatrix
using Profile

rx, rz = 1e-7, 1.3e-7
n_max = 3
k1 = Complex(6.2e6)
k2 = 7e6 + 1e3 * im
rotationally_symmetric = true

calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2; rotationally_symmetric=rotationally_symmetric)


"""
using VectorSphericalWaves
n_θ_points, n_ϕ_points = 10000,20000
θ_array, ϕ_array = Tmatrix.meshgrid_θ_ϕ(n_θ_points, n_ϕ_points; min_θ=1e-16, min_ϕ=1e-16, rotationally_symmetric=rotationally_symmetric) 
r_array, n̂_array = Tmatrix.ellipsoid(rx, rz, θ_array);
VectorSphericalWaves.N_mn_wave_SVector.(0,1,k2*r_array, θ_array,ϕ_array; kind= "regular")
VectorSphericalWaves.N_mn_wave_SVector(0,1,k2*r_array[1], θ_array[1],ϕ_array[1]; kind= "regular")
@time for i = 1:1e3; VectorSphericalWaves.N_mn_wave_SVector.(0,1,k2*r_array, θ_array,ϕ_array; kind= "regular"); end
@time for i = 1:1e3; VectorSphericalWaves.N_mn_wave.(0,1,k2*r_array, θ_array,ϕ_array; kind= "regular"); end
"""