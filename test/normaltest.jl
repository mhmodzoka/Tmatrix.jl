#commented psuedocode

function calculate_Tmatrix_for_any(
    face_list::Vector{Any} #make this better typed
    point_list::Vector{Vector{Float64}}
    n_max::Int,
    k1::Complex{R}, k2::Complex{R};
    HDF5_filename=nothing,
    rotationally_symmetric=false, symmetric_about_plane_perpendicular_z=false,
    BigFloat_precision = nothing
) where {R <: Real}    

    #polar_list = convert_to_polar(point_list) #format is [θ, ϕ, r]
    θ_array, ϕ_array, r_array = #unzip polar_list 

    # calculate r and n̂ for the geometry
    n̂_array = point_list #copying so that size is same

    for element in n̂_array
        element = [0,0,0]
    end

    for face in face_list:
        for (vertex, index) in enumerate(face)
            n_array[vertex] += cross_product(vertex, face[(index+1)%size(face)])
        end
    end

    for element in n̂_array
        element = normalize(element)
    end

    end

    # calculate T-matrix
    k1r_array = k1 .* r_array;
    k2r_array = k2 .* r_array;
    T = T_matrix(
        n_max, k1, k2, k1r_array, k2r_array, r_array, θ_array, ϕ_array, n̂_array;
        HDF5_filename=HDF5_filename, rotationally_symmetric=rotationally_symmetric, symmetric_about_plane_perpendicular_z=symmetric_about_plane_perpendicular_z,
        BigFloat_precision = BigFloat_precision
    )
    return T
end

