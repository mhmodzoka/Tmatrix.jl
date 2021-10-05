using Tmatrix
using Zygote

rx, rz = 1e-6, 1e-6
n_max = 1
k1_r = 1e7;
k1_i = 0.0;
k2_r = 1.5e7;
k2_i = 1e3;

k1 = Complex(k1_r, k1_i)
k2 = Complex(k2_r, k2_i)

n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array = rx * ones(size(θ_array))
ϕ_array = zeros(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

# testing out T-matrix calculation and automatic differentiation using <<T_matrix_SeparateRealImag_arbitrary_mesh>>
@time T_sph = calculate_Tmatrix_for_spheroid_SeparateRealImag(
    rx,
    rz,
    n_max,
    k1_r,
    k1_i,
    k2_r,
    k2_i;
    n_θ_points = n_θ_points,
    rotationally_symmetric = true,
    symmetric_about_plane_perpendicular_z = false,
);
println("T-matrix using spheroid <<calculate_Tmatrix_for_spheroid_SeparateRealImag>>")
display(T_sph)

@time T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
    n_max,
    k1_r,
    k1_i,
    k2_r,
    k2_i,
    r_array,
    θ_array,
    ϕ_array,
    rotationally_symmetric,
    symmetric_about_plane_perpendicular_z,
    BigFloat_precision,
);
println("T-matrix using an arbitrary mesh <<T_matrix_SeparateRealImag_arbitrary_mesh>>")
display(T)

@time ∂T = Zygote.jacobian(
    Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh,
    n_max,
    k1_r,
    k1_i,
    k2_r,
    k2_i,
    r_array,
    θ_array,
    ϕ_array,
    rotationally_symmetric,
    symmetric_about_plane_perpendicular_z,
    BigFloat_precision,
)
println("Jacobian for <<T_matrix_SeparateRealImag_arbitrary_mesh>> calcuated by Zygote:")
display(∂T)

# testing gradient descent optimization for <<sin>>
function target_to_be_minimized(t)
    return sin(t)
end

function ∂target_to_be_minimized(t)
    return Zygote.gradient(target_to_be_minimized, t)[1]
end

"""
learning_rate = 0.01
loss_here = 1e6
t_0 = 0.3
∂loss = 0
while loss_here > 0.01
    t_0 -= learning_rate * ∂loss
    loss_here = target_to_be_minimized(t_0)
    ∂loss = ∂target_to_be_minimized(t_0)
    println("loss_here = loss_here, ∂loss = ∂loss")
end
"""

# testing gradient descent optimization for <<T_matrix_SeparateRealImag_arbitrary_mesh>>
# TODO: use an optimization package instead, so that it has better stop criteria and tricks for faster convergence
rx, rz = 1e-6, 1e-6
n_max = 2
k1_r = 1e7;
k1_i = 0.0;
k2_r = 1.5e7;
k2_i = 1e3;

n_θ_points = 20
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array = rx * ones(size(θ_array))
ϕ_array = zeros(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

function objective_function_normal_mesh_density(r_array)
    # println("I am <<objective_function>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")
    ϕ_array = zeros(size(θ_array))
    T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max,
        k1_r,
        k1_i,
        k2_r,
        k2_i,
        (collect(r_array)),
        θ_array,
        ϕ_array,
        rotationally_symmetric,
        symmetric_about_plane_perpendicular_z,
        BigFloat_precision,
    )
    # return abs(T[1]) # if we need to minimize the first element of T-matrix

    # return sum(abs.(T)) # if we need to minimize the sum of all real and imag

    # return Tmatrix.get_OrentationAv_scattering_CrossSection_from_Tmatrix(T, Complex(k1_r, k1_i)) # if we need to minimize scattering cross section

    return 1 * Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(
        T,
        Complex(k1_r, k1_i),
        Tmatrix.calculate_surface_area_of_axisymmetric_particle(r_array, θ_array),
    ) # if we need to minimize emissivity
end

function objective_function_more_mesh_density(r_array_input)
    r_θ_array = Tmatrix.double_mesh_density(r_array_input, θ_array)
    r_array_double_density = r_θ_array[:, 1]
    θ_array_double_density = r_θ_array[:, 2]
    ϕ_array_double_density = zeros(size(θ_array_double_density))

    # println("I am <<objective_function>>, and the type of r_array_double_density is $(typeof(r_array_double_density))")
    println("r_array_input = $r_array_input")
    println("r_array_double_density = $r_array_double_density")

    T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max,
        k1_r,
        k1_i,
        k2_r,
        k2_i,
        (collect(r_array_double_density)),
        θ_array_double_density,
        ϕ_array_double_density,
        rotationally_symmetric,
        symmetric_about_plane_perpendicular_z,
        BigFloat_precision,
    )
    # return abs(T[1]) # if we need to minimize the first element of T-matrix

    # return sum(abs.(T)) # if we need to minimize the sum of all real and imag

    # return Tmatrix.get_OrentationAv_scattering_CrossSection_from_Tmatrix(T, Complex(k1_r, k1_i)) # if we need to minimize scattering cross section

    return 1 * Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(
        T,
        Complex(k1_r, k1_i),
        Tmatrix.calculate_surface_area_of_axisymmetric_particle(
            r_array_double_density,
            θ_array_double_density,
        ),
    )
end

function objective_function_more_mesh_density_using_Eps(r_array_input)
    if mesh_density_boost == "double"
        r_θ_array = Tmatrix.double_mesh_density(r_array_input, θ_array)
    elseif mesh_density_boost == "quadruple"
        r_θ_array = Tmatrix.quadruple_mesh_density(r_array_input, θ_array)
    elseif mesh_density_boost == "octuple"
        r_θ_array = Tmatrix.octuple_mesh_density(r_array_input, θ_array)
    elseif mesh_density_boost == "sexdecuple"
        r_θ_array = Tmatrix.sexdecuple_mesh_density(r_array_input, θ_array)
    else
        r_θ_array = r_array_input
    end

    r_array_double_density = r_θ_array[:, 1]
    θ_array_double_density = r_θ_array[:, 2]
    ϕ_array_double_density = zeros(size(θ_array_double_density))

    # println("I am <<objective_function>>, and the type of r_array_double_density is $(typeof(r_array_double_density))")
    println("r_array_input = $r_array_input")
    println("r_array_double_density = $r_array_double_density")

    T = Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
        n_max,
        wl_or_freq_input,
        input_unit,
        Eps_r_r_1,
        Eps_r_i_1,
        Mu_r_r_1,
        Mu_r_i_1,
        Eps_r_r_2,
        Eps_r_i_2,
        Mu_r_r_2,
        Mu_r_i_2,
        (collect(r_array_double_density)),
        θ_array_double_density,
        ϕ_array_double_density,
        rotationally_symmetric,
        symmetric_about_plane_perpendicular_z,
        BigFloat_precision,
    )
    # return abs(T[1]) # if we need to minimize the first element of T-matrix

    # return sum(abs.(T)) # if we need to minimize the sum of all real and imag

    # return Tmatrix.get_OrentationAv_scattering_CrossSection_from_Tmatrix(T, Complex(k1_r, k1_i)) # if we need to minimize scattering cross section
    if optical_property_tobe_optimized == "emissivity"
        k1_complex = Tmatrix.get_WaveVector(
            wl_or_freq_input;
            input_unit = input_unit,
            Eps_r = Complex(Eps_r_r_1, Eps_r_i_1),
            Mu_r = Complex(Mu_r_r_1, Mu_r_i_1),
        )

        return 1 * Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(
            T,
            k1_complex,
            Tmatrix.calculate_surface_area_of_axisymmetric_particle(
                r_array_double_density,
                θ_array_double_density,
            ),
        ) # if we need to minimize emissivity
    elseif optical_property_tobe_optimized == "scattering_CrossSection"
        return Tmatrix.get_OrentationAv_scattering_CrossSection_from_Tmatrix(
            T,
            Complex(k1_r, k1_i),
        ) # if we need to minimize scattering cross section
    end
end

# objective_function = objective_function_normal_mesh_density
# objective_function = objective_function_more_mesh_density
objective_function = objective_function_more_mesh_density_using_Eps

function ∂objective_function(r_array)
    # println("I am <<∂objective_function>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")
    return Zygote.gradient(objective_function, (collect(r_array)))[1]
end

using Plots

wl_or_freq_input = 1e-6;
input_unit = "m";
n_max = 1
Eps_r_r_1 = 1.0;
Eps_r_i_1 = 0.0;
Eps_r_r_2 = 1.5;
Eps_r_i_2 = 0.01;
Mu_r_r_1 = 1.0;
Mu_r_i_1 = 0.0;
Mu_r_r_2 = 1.0;
Mu_r_i_2 = 0.0;

optical_property_tobe_optimized = "scattering_CrossSection" # can be any member of this list: ["emissivity", "scattering_CrossSection"]
maximize_or_minimize = "maximize"
mesh_density_boost = "quadruple" # by how many we need to add more points between each two consecutive points defining the particle surface. Can be any of ["double", "quadruple", "octuple", "sexdecuple"]
n_max = 1
n_θ_points = 5
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array_initial, _ = Tmatrix.ellipsoid(0.1e-7, 0.11e-7, θ_array)
# r_array_initial = 0.1e-7 * ones(size(θ_array))
r_array = r_array_initial
ϕ_array = zeros(size(θ_array))
xyz_initial = vcat(
    Tmatrix.convert_coordinates_Sph2Cart.(
        r_array_initial,
        θ_array,
        zeros(size(r_array_initial)),
    )...,
)
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

learning_rate = 0.5e-1
# learning_rate = 1e-2
global loss_here = -1e6
global ∂loss = 0
println("Starting optimization for T-matrix")
global n_iteration = 0
loss_array = []
loss_old = 9999999999999

directory_name_to_save_plots = "cache/iteration_particle_plots/$(maximize_or_minimize)_$(optical_property_tobe_optimized)_07_27_2021_$(mesh_density_boost)_mesh_density_learning_rate_$(learning_rate)__D_0.2e-7_nmax_$(n_max)"
mkpath(directory_name_to_save_plots)

while (n_iteration < 500) # (loss_here < 0)
    n_iteration += 1

    if maximize_or_minimize == "maximize"
        r_array = r_array .+ learning_rate .* ∂loss
    else
        r_array = r_array .- learning_rate .* ∂loss
    end

    loss_here = objective_function(r_array)
    ∂loss = ∂objective_function(r_array)
    append!(loss_array, loss_here)

    println()
    println("iteration #$n_iteration: loss_here = $loss_here, ∂loss = $∂loss")
    println("r_array = $r_array")

    println()
    println("loss_array=$loss_array")
    xyz = vcat(
        Tmatrix.convert_coordinates_Sph2Cart.(r_array, θ_array, zeros(size(r_array)))...,
    )
    p1 = plot(
        xyz_initial[:, 1],
        xyz_initial[:, 3],
        aspect_ratio = :equal,
        label = "initial particle",
    )
    p1 = plot!(
        xyz[:, 1],
        xyz[:, 3],
        aspect_ratio = :equal,
        label = "particle after $n_iteration iterations",
        title = "scattering cross section = $loss_here m",
    )
    p2 = plot(
        1:length(loss_array),
        loss_array,
        xlabel = "iteration #",
        ylabel = "scattering cross section (m)",
    )
    fig = plot(p1, p2, layout = (1, 2), size = (1200, 800))

    figure_name =
        joinpath(directory_name_to_save_plots, "particle_geom_iteration_$n_iteration.png")

    savefig(fig, figure_name)
    # if (loss_here - loss_old) < 1e-6; break; end
    loss_old = loss_here
end

# # trying JuMP with cos
# using JuMP
# using Ipopt
# model = Model(Ipopt.Optimizer)
# @variable(model, x >= 0)
# # @NLconstraint(model, sin(x)>=0)
# g(x::R) where {R <: Real} = cos(x)
# register(model, :g, 1, g; autodiff = true) # try registerNLFunction
# @NLobjective(model, Min, g(x))
# optimize!(model)
# 
# ####################################################################
# # trying JuMP with objective_function
# function objective_function__(r_array...)
#     return objective_function(collect((r_array)))
# end
# function ∂objective_function__(r_array...)
#     return ∂objective_function(collect((r_array)))
# end
# 
# using JuMP
# using Ipopt
# 
# n_θ_points = 10
# θ_array = collect(LinRange(1e-6, pi, n_θ_points))
# r_array = rx * ones(size(θ_array))
# ϕ_array = zeros(size(θ_array))
# rotationally_symmetric = true
# symmetric_about_plane_perpendicular_z = false
# BigFloat_precision = nothing
# function zz()
#     model = Model(Ipopt.Optimizer)
#     # @variable(model, r[1:n_θ_points], lowerbound=zeros(n_θ_points), upperbound=1e-6*ones(n_θ_points))
#     @variable(model, r[1:n_θ_points])
#     for r_here in r
#         JuMP.set_lower_bound(r_here, 1e-7)
#         JuMP.set_upper_bound(r_here, 1e-6)
#     end
#     set_start_value.(r, 0.5e-7 * ones(n_θ_points))
#     # @NLconstraint(model, sin(x)>=0)
#     # g(x::R) where R <: Real = cos(x)
#     register(
#         model,
#         :objective_function__,
#         n_θ_points,
#         objective_function__,
#         ∂objective_function__;
#         autodiff = true,
#     )
#     @NLobjective(model, Min, objective_function__(r...))
#     optimize!(model)
# 
#     return
# end
# zz()
# 
# using Interpolations
# θ_array = collect((0:0.25:1) * pi)
# r_array_initial = 1e-8 .* ones(size(θ_array))
# θ_array_interp = collect(LinRange(0, pi, 10))
# function try_(r_array)
#     """
#     sitp = scale(
#         interpolate(r_array, BSpline(Quadratic(Reflect(OnCell())))),
#         θ_array
#     );
#     r_array_interp = (sitp(θ_array_interp))
#     """
# 
#     """
#     III = BSplineInterpolation(r_array,θ_array,3,:ArcLen,:Average)
#     r_array_interp = III.(θ_array_interp)
#     """
#     return sum(r_array_interp)
# end
# 
# function interpolate_curve(r_array)
#     sitp = scale(interpolate(r_array, BSpline(Quadratic(Reflect(OnCell())))), θ_array)
#     return (sitp(θ_array_interp))
# end

