import numpy as np

def T_matrix(
    n_max: int,
    k1: C, k2: C,
    k1r_array: np.ndarray(dtype=np.complex64), k2r_array: np.ndarray(dtype=np.complex64), # C
    r_array: np.ndarray(dtype=np.real), θ_array: np.ndarray(dtype=np.real), 
    ϕ_array: np.ndarray(dtype=np.real), n̂_array, #TODO normally "n_array::Any;"" in julia, that signals it's required right?
    rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False, HDF5_filename=None,
    verbose=False, create_new_arrays=False, BigFloat_precision = None
):
    if BigFloat_precision != None:
        def T_matrix_BigFloat(n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
                HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,  
                ):
            return T_matrix(
                n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
                HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,                
        return setprecision(T_matrix_BigFloat, BigFloat_precision)    #I think this is how do blocks translate
        
    else:
        if create_new_arrays
            #args are keyword by deafult in python so I'm leaving it

            RgQ =       Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;
            kind="regular" , rotationally_symmetric=rotationally_symmetric, 
            symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)

            Q_inv = inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array; 
            kind="irregular", rotationally_symmetric=rotationally_symmetric, 
            symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))

            T = -1 .* RgQ * Q_inv

        else
            T = (
                -     Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="regular"  , rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)
                * inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;kind="irregular", rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))
            )
        

        if HDF5_filename != None
            save_Tmatrix_to_HDF5_file(T, HDF5_filename)
        

        return T

def calculate_Tmatrix_for_spheroid(
        rx: R, rz: R, n_max: int,
        k1: Complex{R}, k2: Complex{R};
        n_θ_points=10, n_ϕ_points=20, HDF5_filename=None,
        rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False,
        BigFloat_precision = None
    ) where {R <: Real}    
    
    # create a grid of θ_ϕ
    θ_array, ϕ_array = meshgrid_θ_ϕ(n_θ_points, n_ϕ_points; min_θ=1e-16, min_ϕ=1e-16, rotationally_symmetric=rotationally_symmetric)    
    
    # calculate r and n̂ for the geometry
    r_array, n̂_array = ellipsoid(rx, rz, θ_array);

    # calculate T-matrix
    k1r_array = k1 .* r_array;
    k2r_array = k2 .* r_array;
    T = T_matrix(
        n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;
        HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,
        BigFloat_precision = BigFloat_precision
    )
    return T

