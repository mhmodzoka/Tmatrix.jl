using Tmatrix
import Zygote

# inputs
rx, rz = 1e-6, 1.3e-6
n_max = 1
k1_r = 1e7; k1_i = 0.0
k2_r = 1.5e7; k2_i = 1e3

calculate_Tmatrix_for_spheroid_SeparateRealImag(rx, rz, n_max, k1_r, k1_i, k2_r, k2_i; rotationally_symmetric=false, symmetric_about_plane_perpendicular_z=false);

function Tmatrix_spheroid_simple(rx, rz)
    return calculate_Tmatrix_for_spheroid_SeparateRealImag(rx, rz, n_max, k1_r, k1_i, k2_r, k2_i; rotationally_symmetric=false, symmetric_about_plane_perpendicular_z=false);
end

Zygote.jacobian(Tmatrix_spheroid_simple, rx, rz)