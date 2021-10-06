#commented psuedocode
using Meshes
using StaticArrays
using LinearAlgebra
using CoordinateTransformations
using Tmatrix

function calculate_Tmatrix_for_any(
    face_list::Vector{Any}, #make this better typed
    point_list::Vector{Vector{Float64}},
    n_max::Int,
    k1::Complex{R},
    k2::Complex{R};
    rotationally_symmetric = false,
    symmetric_about_plane_perpendicular_z = false,
    BigFloat_precision = nothing,
) where {R <: Real}

    #spherical_list = convert_to_spherical(point_list) #format is [r, θ, ϕ], use coordinatetransformations.jl to convert

    #TODO: bundle these into a new struct and make that not break
    r_array, θ_array, ϕ_array = (
        [element[1] for element in spherical_list],
        [element[2] for element in spherical_list],
        [element[3] for element in spherical_list],
    )

    # calculate r and n̂ for the geometry

    n̂_array = point_list #copying so that size is same

    for element in n̂_array
        element = [0, 0, 0]
    end

    for face in face_list
        for (vertex, i) in enumerate(face)
            n_array[vertex] += vertex × face[(i + 1) % size(face)]
        end
    end

    for element in n̂_array
        element = normalize(element) #TODO make sure this works, use normalize!(element) later if at all
    end

    # calculate T-matrix
    k1r_array = k1 .* r_array
    k2r_array = k2 .* r_array
    T = T_matrix(
        n_max,
        k1,
        k2,
        k1r_array,
        k2r_array,
        r_array,
        θ_array,
        ϕ_array,
        n̂_array;
        HDF5_filename = HDF5_filename,
        rotationally_symmetric = rotationally_symmetric,
        symmetric_about_plane_perpendicular_z = symmetric_about_plane_perpendicular_z,
        BigFloat_precision = BigFloat_precision,
    )
    return T
end

point_list = Point(3, 8, 23), Point(1, 2, 3), Point(5, 12, 12), Point(15, 3, 7)

face_list = [connect((1, 2, 3)), connect((1, 2, 4)), connect((1, 3, 4)), connect((2, 3, 4))]

#convert_to_spherical

#TODO make generic over shape
function Spherical(point_coords::Point)
    Spherical(point_coords.coords[1], point_coords.coords[2], point_coords.coords[3])
end

polar_point_list = [Spherical(p) for p in point_list]

n_max = 1
k1_r, k1_i, k2_r, k2_i = 1e5, 1e3, 2e5, 3e3

calculate_Tmatrix_for_spheroid_SeparateRealImag(
    face_list,
    point_list,
    n_max,
    k1_r,
    k1_i,
    k2_r,
    k2_i;
    rotationally_symmetric = false,
    symmetric_about_plane_perpendicular_z = false,
);

