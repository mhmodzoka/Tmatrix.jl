# when Tmatrix size grows, I can see error in inverting Q_matrix
using Tmatrix
using DataStructures

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 15
k1_r = 1e7;
k1_i = 0.0;
k2_r = 1.5e7;
k2_i = 1e3;

k1 = Complex(k1_r, k1_i)
k2 = Complex(k2_r, k2_i)

T = SortedDict()

kk = 10
for nn in 2:n_max
    # calculating Tmatrix for different sizes
    println()
    @time T[nn] = calculate_Tmatrix_for_spheroid(
        rx,
        rz,
        nn,
        k1,
        k2;
        n_θ_points = 100,
        rotationally_symmetric = false,
        symmetric_about_plane_perpendicular_z = false,
        BigFloat_precision = nothing,
    )
    println("n_max = $nn, matrix to be inverted has size: $(size(T[nn]))")
    println(
        "max error compared with n_max = 2: $(max(abs.(T[nn][1:kk ,1:kk] - T[2][1:kk ,1:kk])...))",
    )
    display(T[nn][1:kk, 1:kk])
end
