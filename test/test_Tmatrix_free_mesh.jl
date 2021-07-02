using Tmatrix
using Zygote

rx, rz = 1e-6, 1e-6
n_max = 1
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3

k1 = Complex(k1_r, k1_i)
k2 = Complex(k2_r, k2_i)

n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array = rx * ones(size(θ_array))
ϕ_array = zeros(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

@time T_sph = calculate_Tmatrix_for_spheroid_SeparateRealImag(
    rx, rz, n_max, k1_r, k1_i, k2_r, k2_i;
    n_θ_points = n_θ_points,
    rotationally_symmetric=true, symmetric_about_plane_perpendicular_z=false
);
println("T-matrix using spheroid <<calculate_Tmatrix_for_spheroid_SeparateRealImag>>")
display(T_sph)

@time T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max, k1_r, k1_i, k2_r, k2_i,
        r_array, θ_array, ϕ_array,
        rotationally_symmetric, symmetric_about_plane_perpendicular_z, BigFloat_precision
    );
println("T-matrix using an arbitrary mesh <<T_matrix_SeparateRealImag_arbitrary_mesh>>")
display(T)
"""
@time ∂T = Zygote.jacobian(
    Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh,
    n_max, k1_r, k1_i, k2_r, k2_i,
    r_array, θ_array, ϕ_array,
    rotationally_symmetric, symmetric_about_plane_perpendicular_z, BigFloat_precision
)
"""



function target_to_be_minimized(t)
    return sin(t)
end

function ∂target_to_be_minimized(t)
    return Zygote.gradient(target_to_be_minimized, t)[1]
end



learning_rate = 0.01
loss_here = 1e6
t_0 = 0.3
∂loss = 0
while loss_here > 0.01
    t_0 = t_0 - learning_rate*∂loss
    loss_here = target_to_be_minimized(t_0)
    ∂loss = ∂target_to_be_minimized(t_0)    
    println("loss_here = $loss_here, ∂loss = $∂loss")
end













rx, rz = 1e-6, 1e-6
n_max = 1
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3

n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array = rx * ones(size(θ_array))
ϕ_array = zeros(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

@time T_sph = calculate_Tmatrix_for_spheroid_SeparateRealImag(
    rx, rz, n_max, k1_r, k1_i, k2_r, k2_i;
    n_θ_points = n_θ_points,
    rotationally_symmetric=true, symmetric_about_plane_perpendicular_z=false
);


function some_target_to_minimize_for_input_Tmatrix(r_array)
    T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max, k1_r, k1_i, k2_r, k2_i,
        r_array, θ_array, ϕ_array,
        rotationally_symmetric, symmetric_about_plane_perpendicular_z, BigFloat_precision
    )
    # return abs(T[1]) # if we need to minimize the first element of T-matrix
    # return sum(abs.(T)) # if we need to minimize the sum of all real and imag
    return get_OrentationAv_scattering_CrossSections_from_Tmatrix(T, Complex(k1_r, k1_i)) # if we need to minimize scattering cross section
end

function ∂some_target_to_minimize_for_input_Tmatrix(T)
    return Zygote.gradient(some_target_to_minimize_for_input_Tmatrix, r_array)[1]
end

using Plots
n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array_initial = 1e-7 * ones(size(θ_array))
r_array = r_array_initial
ϕ_array = zeros(size(θ_array))
xyz_initial = vcat(Tmatrix.convert_coordinates_Sph2Cart.(r_array_initial, θ_array, zeros(size(r_array_initial)))...)
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

learning_rate = 0.5*1e-8
learning_rate = 1e-2
loss_here = 1e6
∂loss = 0
println("Starting optimization for T-matrix")
n_iteration = 0
loss_array = []
loss_old = 9999999999999
while (loss_here > 1e-25) & (n_iteration < 250)
    n_iteration += 1
    r_array = r_array .- learning_rate.*∂loss
    loss_here = some_target_to_minimize_for_input_Tmatrix(r_array)
    ∂loss = ∂some_target_to_minimize_for_input_Tmatrix(r_array)   
    append!(loss_array, loss_here) 
    println("iteration #$n_iteration: loss_here = $loss_here, ∂loss = $∂loss")
    xyz = vcat(Tmatrix.convert_coordinates_Sph2Cart.(r_array, θ_array, zeros(size(r_array)))...)
    p1 = plot(xyz_initial[:,1], xyz_initial[:,3], aspect_ratio=:equal, label="initial particle")
    p1 = plot!(xyz[:,1], xyz[:,3], aspect_ratio=:equal, label="particle after $n_iteration iterations", title="scattering cross section = $loss_here m")
    p2 = plot(1:length(loss_array), loss_array, xlabel="iteration #", ylabel="scattering cross section (m)")
    fig = plot(p1,p2,layout = (1, 2), size=(1200,800))
    savefig(
        fig,
        "cache/iteration_particle_plots/minimizing_scattering_cross_section_5/particle_geom_iteration_$n_iteration.png"
    )
    #if (loss_here - loss_old) < 1e-6; break; end
    loss_old = loss_here
end

using Plots
for x = 1:10
    plot([1,2,3], x.*[1,2,3])
    sleep(2)
end



using ComplexOperations
function get_OrentationAv_scattering_CrossSections_from_Tmatrix(T, k1)
    if size(T)[1] == size(T)[2] # if T-matrix is a square matrix, then T-matrix is complex
        return 2*pi/k1^2 * sum(T .* conj(T))
    elseif size(T)[1] == size(T)[2]/2 # if T-matrix has number of columns double the number of rows, then T-matrix is hcat() of real and imag parts of Tmatrix        
        return get_OrentationAv_scattering_CrossSections_from_Tmatrix_SeparateRealImag(T, real(k1), imag(k1))
    end    
end

function get_OrentationAv_scattering_CrossSections_from_Tmatrix_SeparateRealImag(T, k1_r, k1_i)
    T_r,T_i = Tmatrix.separate_real_imag(T)
    T_by_conjT_sum = sum(complex_multiply.(T_r, T_i, T_r, -T_i))
    k1_squared = complex_multiply(k1_r, k1_i, k1_r, k1_i)
    two_pi_over_k1_squared = complex_divide(2*pi, 0, k1_squared[1], k1_squared[2])
    return complex_multiply(two_pi_over_k1_squared[1], two_pi_over_k1_squared[2], T_by_conjT_sum[1], T_by_conjT_sum[2])[1]    
end 


T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max, k1_r, k1_i, k2_r, k2_i,
        r_array, θ_array, ϕ_array,
        rotationally_symmetric, symmetric_about_plane_perpendicular_z, BigFloat_precision
    )

get_OrentationAv_scattering_CrossSections_from_Tmatrix(T, Complex(k1_r,k1_i))