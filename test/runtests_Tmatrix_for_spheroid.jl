# %%
using Tmatrix

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 3
k1_r = 1e7;
k1_i = 0.0;
k2_r = 1.5e7;
k2_i = 1e3;

k1 = Complex(k1_r, k1_i)
k2 = Complex(k2_r, k2_i)

BigFloat_precision_list = 2 .^ [6, 7, 8, 9]

println("***************************************************************")
println("***************************************************************")
println("Calculating T-matrix using Complex numbers")
println("***************************************************************")
println("***************************************************************")

println("using Float64 =====================================")
println("No symmetry exploited:")
@time T_no_symmetry_float_64 = calculate_Tmatrix_for_spheroid(
    rx,
    rz,
    n_max,
    k1,
    k2,
    rotationally_symmetric = false,
    symmetric_about_plane_perpendicular_z = false,
);

# %%
println("exploiting symmetry:")
@time T_with_symmetry_float_64 = calculate_Tmatrix_for_spheroid(
    rx,
    rz,
    n_max,
    k1,
    k2,
    rotationally_symmetric = true,
    symmetric_about_plane_perpendicular_z = true,
);

println("using BigFloat =====================================")
println("exploiting symmetry:")
T_with_symmetry_big_float = []
for precision_here in BigFloat_precision_list
    println("using precision $precision_here")
    @time T = calculate_Tmatrix_for_spheroid(
        rx,
        rz,
        n_max,
        k1,
        k2,
        rotationally_symmetric = true,
        symmetric_about_plane_perpendicular_z = true,
        BigFloat_precision = precision_here,
    )
    append!(T_with_symmetry_big_float, T)
    println()
end

"""
n_array = collect(1:10)
elapsed_time = zeros(size(n_array))
for n = 1:length(n_array)
elapsed_time[n] = @elapsed calculate_Tmatrix_for_spheroid(rx, rz, n_max, k1, k2, rotationally_symmetric=true, symmetric_about_plane_perpendicular_z=true, BigFloat_precision=8)
end
"""
println()
println()
println()
println("***************************************************************")
println("***************************************************************")
println("Calculating T-matrix using real numbers")
println("***************************************************************")
println("***************************************************************")

println("using Float64 =====================================")
println("No symmetry exploited:")
@time T_no_symmetry_float_64_SeparateRealImag =
    calculate_Tmatrix_for_spheroid_SeparateRealImag(
        rx,
        rz,
        n_max,
        k1_r,
        k1_i,
        k2_r,
        k2_i;
        rotationally_symmetric = false,
        symmetric_about_plane_perpendicular_z = false,
    );

println("exploiting symmetry:")
@time T_with_symmetry_float_64_SeparateRealImag =
    calculate_Tmatrix_for_spheroid_SeparateRealImag(
        rx,
        rz,
        n_max,
        k1_r,
        k1_i,
        k2_r,
        k2_i;
        rotationally_symmetric = true,
        symmetric_about_plane_perpendicular_z = true,
    );

println("using BigFloat =====================================")
println("exploiting symmetry:")
T_with_symmetry_big_float_SeparateRealImag = []
for precision_here in BigFloat_precision_list
    println("using precision $precision_here")
    @time T = calculate_Tmatrix_for_spheroid_SeparateRealImag(
        rx,
        rz,
        n_max,
        k1_r,
        k1_i,
        k2_r,
        k2_i;
        rotationally_symmetric = true,
        symmetric_about_plane_perpendicular_z = true,
        BigFloat_precision = precision_here,
    )
    append!(T_with_symmetry_big_float_SeparateRealImag, T)
end

