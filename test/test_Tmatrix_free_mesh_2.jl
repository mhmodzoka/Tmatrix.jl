using Tmatrix
using Zygote
using JuMP
using Ipopt

function some_target_to_minimize_for_input_Tmatrix(r_array)
    # println(r_array)
    # println("I am <<some_target_to_minimize_for_input_Tmatrix>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")

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

function ∂some_target_to_minimize_for_input_Tmatrix(r_array)
    # println("I am <<∂some_target_to_minimize_for_input_Tmatrix>>, and the type of r_array is $(typeof(r_array))")
    # println("r_array = $r_array")
    return Zygote.gradient(some_target_to_minimize_for_input_Tmatrix, (collect(r_array)))[1]
end

rx, rz = 1e-6, 1e-6
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

####################################################################
# trying JuMP with some_target_to_minimize_for_input_Tmatrix
function some_target_to_minimize_for_input_Tmatrix__(r_array...)
    # println("this is <<some_target_to_minimize_for_input_Tmatrix__>>, collect(r_array) = $(collect(r_array))")
    # println("r_array = $r_array")

    return some_target_to_minimize_for_input_Tmatrix(collect((r_array)))
end
function ∂some_target_to_minimize_for_input_Tmatrix__(r_array...)
    # println("this is <<∂some_target_to_minimize_for_input_Tmatrix__>>, collect(r_array) = $(collect(r_array))")
    # println("r_array = $r_array")
    # println("r_array[1] = $(r_array[1])")
    # println("r_array[2:end] = $(r_array[2:end])")

    return ∂some_target_to_minimize_for_input_Tmatrix(collect((r_array[2:end])))
end

n_θ_points = 10
θ_array = collect(LinRange(1e-6, pi, n_θ_points))
r_array = rx * ones(size(θ_array))
ϕ_array = zeros(size(θ_array))
rotationally_symmetric = true
symmetric_about_plane_perpendicular_z = false
BigFloat_precision = nothing

model = Model(Ipopt.Optimizer)
set_optimizer_attributes(model, "tol" => 1e-16, "max_iter" => 10)
set_time_limit_sec(model, 3000)
@variable(model, r[1:n_θ_points])
for r_here in r
    JuMP.set_lower_bound(r_here, 0.1e-7)
    JuMP.set_upper_bound(r_here, 5e-6)
    JuMP.set_start_value(r_here, 4e-6)
end
# set_start_value.(r, 0.5e-7*ones(n_θ_points).+rand(n_θ_points)*1e-8)
JuMP.register(
    model,
    :some_target_to_minimize_for_input_Tmatrix__,
    n_θ_points,
    some_target_to_minimize_for_input_Tmatrix__,
    ∂some_target_to_minimize_for_input_Tmatrix__,
)
@NLobjective(model, Max, some_target_to_minimize_for_input_Tmatrix__(r...))
optimize!(model)
println("optimum r is $(value.(r))")
