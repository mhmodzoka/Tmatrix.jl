using Zygote
using Tmatrix
import Tmatrix
using VectorSphericalWaves

# some objective function to work with
k1, k2 = 1e6, 1.5e6
n_θ_points = 100
n_max = 2

m, n = 1, 2
m_, n_ = 1, 2

rx, rz = 1e-6, 1.3e-6
rotationally_symmetric = true; use_Alok_vector_preallocation = true




function f(ab)
    # T = calculate_Tmatrix_for_spheroid(ab[1], ab[2], n_max, k1, k2; n_θ_points=n_θ_points, rotationally_symmetric=true, symmetric_about_plan_perpendicular_z=true)
    # T = Tmatrix.Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array; use_Alok_vector_preallocation=true, rotationally_symmetric=true)
    # T = Tmatrix.J_mn_m_n_(m, n, m_, n_, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array; use_Alok_vector_preallocation=true, rotationally_symmetric=true)
    # return real(sum(T))    

    # try VSWF
    rx, rz = ab
    θ_1D_array = LinRange(1e-16, π, n_θ_points);
    ϕ_1D_array = LinRange(1e-16, 2π, 2 * n_θ_points);
    if rotationally_symmetric        
        θ_array = θ_1D_array
        ϕ_array = zeros(size(θ_array))
    else
        # eventually, this should be removed. I just keep it for sanity checks.
        θ_array, ϕ_array = Tmatrix.meshgrid(θ_1D_array, ϕ_1D_array);
    end
    r_array, n̂_array = Tmatrix.ellipsoid(rx, rz, θ_array; use_Alok_vector_preallocation=use_Alok_vector_preallocation);
    k1r_array = k1 .* r_array;
    k2r_array = k2 .* r_array;
    M = M_mn_wave.(m, n, k1r_array, θ_array, ϕ_array; kind="regular")
    # M = Tmatrix.M_mn_wave_array(m_, n_, k2r_array, θ_array, ϕ_array, kind = kind_first_function, use_Alok_vector_preallocation = use_Alok_vector_preallocation)    
    return imag(sum(sum(M)))    
end

f'([rx,rz])