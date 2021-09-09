# %%
using Tmatrix

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 3
k1_r = 1e7;
k1_i = 0.0;
k2_r = 1.5e7;
k2_i = 1e3;

n_θ_points = 10
n_ϕ_points = 20

rotationally_symmetric = false

θ_array, ϕ_array = meshgrid_θ_ϕ(
    n_θ_points,
    n_ϕ_points;
    min_θ = 1e-16,
    min_ϕ = 1e-16,
    rotationally_symmetric = rotationally_symmetric,
)

r_array, n̂_array = ellipsoid(rx, rz, θ_array)
