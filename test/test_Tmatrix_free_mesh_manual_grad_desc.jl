using Tmatrix
using Zygote
using Plots
function objective_function(r_array_input, θ_array_input)
    # println(r_array)
    # println("I am <<objective_function>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")

    # quadrupling_mesh
    r_θ_array = Tmatrix.quadruple_mesh_density(r_array_input, θ_array_input)
    r_array = r_θ_array[:, 1]
    θ_array = r_θ_array[:, 2]
    println(r_array)
    println(θ_array)
    ϕ_array = zeros(size(θ_array))

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
        (collect(r_array)),
        θ_array,
        ϕ_array,
        rotationally_symmetric,
        symmetric_about_plane_perpendicular_z,
        BigFloat_precision,
    )
    k1_complex = Tmatrix.get_WaveVector(
        wl_or_freq_input;
        input_unit = input_unit,
        Eps_r = Complex(Eps_r_r_1, Eps_r_i_1),
        Mu_r = Complex(Mu_r_r_1, Mu_r_i_1),
    )
    # return abs(T[1]) # if we need to minimize the first element of T-matrix

    # return sum(abs.(T)) # if we need to minimize the sum of all real and imag

    return Tmatrix.get_OrentationAv_scattering_CrossSection_from_Tmatrix(T, k1_complex) # if we need to minimize scattering cross section

    """
    return -1 * Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(
        T,
        Complex(k1_r, k1_i),
        Tmatrix.calculate_surface_area_of_axisymmetric_particle(r_array, θ_array)
    ) # if we need to minimize emissivity
    """
end

function ∂objective_function(r_array, θ_array)
    # println("I am <<∂objective_function>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")
    return Zygote.gradient(
        r_array -> objective_function(r_array, θ_array),
        (collect(r_array)),
    )[1]
end

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

n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi - 1e-6, n_θ_points))
r_array = 0.5 * wl_or_freq_input * ones(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

learning_rate = 0.5e-4
global loss_here = -1e6
global ∂loss = 0
println("Starting optimization for T-matrix ...")
loss_array = []
loss_old = 9999999999999
for n_iteration in 1:50
    global r_array = r_array .- learning_rate .* ∂loss

    global loss_here = objective_function(r_array, θ_array)
    global ∂loss = ∂objective_function(r_array, θ_array)
    append!(loss_array, loss_here)

    println()
    println("iteration #$n_iteration: loss_here = $loss_here, ∂loss = $∂loss")
    println("r_array = $r_array")

    xyz = vcat(
        Tmatrix.convert_coordinates_Sph2Cart.(r_array, θ_array, zeros(size(r_array)))...,
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
    mkpath("cache/iteration_particle_plots/maximizing_emissivity")
    savefig(
        fig,
        "cache/iteration_particle_plots/maximizing_emissivity/particle_geom_iteration_$(n_iteration).png",
    )
    # if (loss_here - loss_old) < 1e-6; break; end
    global loss_old = loss_here
end

