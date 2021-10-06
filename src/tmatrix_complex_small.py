from numpy import complex128
import jax.numpy as np
from jax.numpy import sin,cos,pi,mgrid,shape
import utils_jax
import vector_spherical_waves_complex as vsw_complex
import VSW_utils as vswu
#use .obj?


def Q_matrix(
        n_max: int,
        k1: np.complex128, k2: np.complex128,
        k1r_array: np.ndarray.dtype, k2r_array: np.ndarray, # C
        r_array: np.ndarray, θ_array: np.ndarray, 
        ϕ_array: np.ndarray, n̂_array, #TODO normally "n_array::Any;"" in julia, that signals it's required right?
        kind="regular", rotationally_symmetric=False, symmetric_about_plane_perpicular_z=False,
        verbose=False,
    ): #where {R <: Real, C <: Complex{R}}
    idx_max = utils_jax.get_max_single_index_from_n_max(n_max)
    Q_mn_m_n_11 = np.zeros((idx_max, idx_max), np.complex128) # TODO: @Alok, should I replace arrays with SMatrix? I am afraid it may get slower, as I have seen that StaticArray may get slower for arrays larger than 100 elements
    Q_mn_m_n_12 = np.zeros((idx_max, idx_max), np.complex128)
    Q_mn_m_n_21 = np.zeros((idx_max, idx_max), np.complex128)
    Q_mn_m_n_22 = np.zeros((idx_max, idx_max), np.complex128)

    idx = 0
    for n in range(1,n_max+1):
        for m in range(n, n+1):
            idx += 1
            idx_ = 0
            for n_ in range(1,n_max+1):
                for m_ in range(-n_, n_+1):
                    idx_ += 1
                    if verbose: print("n,m,idx = $n,$m,$idx  n_,m_,idx_ = $n_,$m_,$idx_")
                    if rotationally_symmetric:
                        if m == m:
                            if symmetric_about_plane_perpicular_z:
                                # apply equations 5.208 and 5.209
                                if (n + n_)%2 == 0:
                                    Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                    Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                                else:
                                    Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                    Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                                
                            else:
                                Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                            
                        
                    else:
                        if symmetric_about_plane_perpicular_z and (m == m_):
                            # apply equations 5.208 and 5.209
                            if (n + n_)%2 == 0:
                                Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                            else:
                                Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                                Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                            
                        else:
                            Q_mn_m_n_11[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=11, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_12[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=12, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_21[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=21, rotationally_symmetric=rotationally_symmetric)
                            Q_mn_m_n_22[idx, idx_] = Q_mn_m_n_(m, n, m_, n_, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,kind=kind,Q_superscript=22, rotationally_symmetric=rotationally_symmetric)
                        
            
        
    
    Q = np.vstack(
        (np.hstack(Q_mn_m_n_11, Q_mn_m_n_12)),
        (np.hstack(Q_mn_m_n_21, Q_mn_m_n_22))
    )
    return Q

        
def Q_mn_m_n_(
        m: int, n: int, m_: int, n_: int,
        k1: np.complex128, k2: np.complex128,
        k1r_array: np.ndarray, k2r_array: np.ndarray, # C
        r_array: np.ndarray, θ_array: np.ndarray, 
        ϕ_array: np.ndarray, n̂_array, #TODO normally "n_array::Any;"" in julia, that signals it's required right?
        kind="regular", Q_superscript=11, rotationally_symmetric=False,
    ): #where {R <: Real, C <: Complex{R}}

    if Q_superscript == 11: J_superscript_1 = 21 ; J_superscript_2 = 12
    elif Q_superscript == 12: J_superscript_1 = 11 ; J_superscript_2 = 22
    elif Q_superscript == 21: J_superscript_1 = 22 ; J_superscript_2 = 11
    elif Q_superscript == 22: J_superscript_1 = 12 ; J_superscript_2 = 21

    Q = (
        -1j * k1 * k2 * J_mn_m_n_(m, n, m_, n_, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array, kind=kind, J_superscript=J_superscript_1, rotationally_symmetric=rotationally_symmetric)
        -1j * k1**2    * J_mn_m_n_(m, n, m_, n_, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array, kind=kind, J_superscript=J_superscript_2, rotationally_symmetric=rotationally_symmetric)
    )

    return Q

def J_mn_m_n_(
        m: int, n: int, m_: int, n_: int,
        k1: np.complex128, k2: np.complex128,
        k1r_array: np.ndarray, k2r_array: np.ndarray, # C
        r_array: np.ndarray, θ_array: np.ndarray, 
        ϕ_array: np.ndarray, n̂_array, #TODO normally "n_array::Any;"" in julia, that signals it's required right?
        kind="regular", J_superscript=11, rotationally_symmetric=False,
    ): #where {R <: Real, C <: Complex{R}}
    
    if rotationally_symmetric:
        # make sure that θ_array is 1D
        if θ_array.ndim != 1:
            print("make this a proper error") #TODO make this a proper error
        ϕ_array = np.zeros(np.shape(θ_array)).astype(type(θ_array))

    # getting the integrand
    if rotationally_symmetric and (m != m_):
        # the integral over ϕ is 2π * δ_m_m_, so it is zero if m != m_
        J_integrand_dS = np.zeros(np.shape(θ_array))
    else:
        J_integrand_dS = J_mn_m_n__integrand(
            m, n,m_,n_,
            k1r_array,k2r_array,
            r_array,θ_array,ϕ_array,n̂_array,
            kind=kind,J_superscript=J_superscript
        )

    # calculate the surface integral
    if rotationally_symmetric:
        # integrate over θ only
        J = 2*pi* np.trapz(J_integrand_dS,(θ_array)) #TODO check that numpy.trapz is equivalent to julia trapz; I think it's just flipped x and y
    else:
        # integrate over θ and ϕ
        # TODO: replace this integral with surface mesh quadrature, like this one: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5095443/ , https://github.com/jareeger/Smooth_Closed_Surface_Quadrature_RBF-julia
        # assuming that θ_array, ϕ_array were created with meshgrid function
        J = np.trapz(J_integrand_dS, (θ_array[:,1], ϕ_array[1,:]))

    return J

def J_mn_m_n__integrand(
    m: int, n: int, m_: int, n_: int,
    k1r_array: np.ndarray, k2r_array: np.ndarray, # C
    r_array: np.ndarray, θ_array: np.ndarray, 
    ϕ_array: np.ndarray, n̂_array,# TODO: I don't know why I get an error when I use n̂_array: AbstractVecOrMat{Vector{Float64}}
    kind="regular", J_superscript=11
): #where {R <: Real, C <: Complex{R}}

    # determining the type of the first the second VSWF
    if J_superscript == 11: # TODO: this if-statement can be done more nicely. We separate J_superscript into two pieces, the number 1 represents M_mn_wave, while number 2 represents vsw_complex.N_mn_wave
        first_function = vsw_complex.M_mn_wave
        second_function = vsw_complex.M_mn_wave
    elif J_superscript == 12:
        first_function = vsw_complex.M_mn_wave
        second_function = vsw_complex.N_mn_wave
    elif J_superscript == 21:
        first_function = vsw_complex.N_mn_wave
        second_function = vsw_complex.M_mn_wave
    elif J_superscript == 22:
        first_function = vsw_complex.N_mn_wave
        second_function = vsw_complex.N_mn_wave
    else:
        print("J_superscript must be any of [11,12,21,22]") #TODO make error


    # determining the type of the first and second VSWF
    kind_first_function = "regular"
    if kind == "irregular":
        kind_second_function = "irregular"
    elif kind == "regular":
        kind_second_function = "regular"
    else:
        print("""kind must be any of ["regular", "irregular"]""") #TODO make error


    # the cross product
    cross_product_MN = np.cross(
        first_function(m_, n_, k2r_array, θ_array, ϕ_array, kind=kind_first_function), # I can directly call M,N waves, with dots.
        second_function(-m, n, k1r_array, θ_array, ϕ_array, kind=kind_second_function))

    # dot multiplying the cross product by the unit vector, and multiplying by (-1)^m
    cross_product_MN_dot_n̂ = (-1)** m * np.dot(cross_product_MN, n̂_array)

    # multiplying by dS=r²sin(θ)
    J_integrand = utils_jax.surface_integrand(cross_product_MN_dot_n̂, r_array, θ_array)

    return J_integrand



def T_matrix(
    n_max: int,
    k1: np.complex128, k2: np.complex128,
    k1r_array: np.ndarray, k2r_array: np.ndarray, # C
    r_array: np.ndarray, θ_array: np.ndarray, 
    ϕ_array: np.ndarray, n̂_array, #TODO normally "n_array::Any;"" in julia, that signals it's required right?
    rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False, HDF5_filename=None,
    verbose=False, create_new_arrays=False, BigFloat_precision = None
):
    # if BigFloat_precision != None:
    #     def T_matrix_BigFloat(n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array],
    #             HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,  
    #             ):
    #         return T_matrix(
    #             n_max, big(k1), big(k2), big.(k1r_array), big.(k2r_array), big.(r_array), big.(θ_array), big.(ϕ_array), [big.(n) for n in n̂_array];
    #             HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,                
    #     return setprecision(T_matrix_BigFloat, BigFloat_precision)    #I think this is how do blocks translate
        
    # else:
    if create_new_arrays:
        #args are keyword by deafult in python so I'm leaving it
        RgQ =       Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,
        kind="regular" , rotationally_symmetric=rotationally_symmetric, 
        symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)

        Q_inv = np.linalg.inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array, 
        kind="irregular", rotationally_symmetric=rotationally_symmetric, 
        symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))

        T = -1 * RgQ * Q_inv

    else:
        T = (
            -     Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array, kind="regular"  , rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose)
            * np.linalg.inv(Q_matrix(n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array, kind="irregular", rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z, verbose=verbose))
        )
    
    # TODO make the HDF5 saving in python
    # if HDF5_filename != None:
    #     save_Tmatrix_to_HDF5_file(T, HDF5_filename)
    

    return T


def calculate_Tmatrix(
        face_list, #TODO make this better typed
        point_list, n_max: int,
        k1: np.complex128, k2: np.complex128,
        n_θ_points=10, n_ϕ_points=20, HDF5_filename=None,
        rotationally_symmetric=False, symmetric_about_plane_perpendicular_z=False,
        BigFloat_precision = None
    ): #where {R <: Real}    
    
    spherical_list = point_list #format is [r, θ, ϕ], use [basic numpy?] to convert
    r_array, θ_array, ϕ_array = ([element[1] for element in spherical_list], [element[2] for element in spherical_list], [element[3] for element in spherical_list])
    # calculate r and n̂ for the geometry

    n̂_array = point_list #copying so that size is same

    for element in n̂_array:
        element = [0,0,0]

    for face in face_list:
        for (vertex, i) in enumerate(face):
            n̂_array[vertex] += np.cross(vertex, face[(i+1)%len(face)])

    for element in n̂_array:
        element = utils_jax.normalize(element) #TODO make sure this works, use normalize!(element) later if at all

    # calculate T-matrix
    k1r_array = k1 * r_array
    k2r_array = k2 * r_array
    T = T_matrix(
        n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array,
        HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,
        BigFloat_precision = BigFloat_precision
    )
    return T