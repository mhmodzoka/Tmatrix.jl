# Contains all functions handling geometry

"""
    Calculate spherical coordinates arrays representing the surface of the particle

This can be either 1D arrays for axi-symmetric particles or 2D arrays for general 3D particles
"""
function get_r_θ_ϕ_arrays(
    geometry;
    angular_resolution = 0.5,
    θ_array = nothing,
    ϕ_array = nothing,
)
    # TODO: use a package that takes a mesh and do integrals
    # check: https://github.com/jareeger/Smooth_Closed_Surface_Quadrature_RBF-julia

    if geometry isa String
        if geometry in ["wire", "cylinder"]
        elseif geometry in ["spheroid", "ellipsoid"]
        end
    elseif geometry isa Function
        # TODO: check if the function has one or two arguments

    else
        throw(DomainError("Please define geometry as 'String', 'Function', or 'Filepath'"))
    end
end

function create_array_from_elements(x1, x2, x3)
    # TODO: @Alok, should I use SVector? what are the drawbacks?
    # return [x1,x2,x3]
    return SVector(x1, x2, x3)
end

"""
    Calculate radial coordinate of ellipsoidal particle surface, given θ points

returns r and n̂ coordinate as a function of θ_array
"""
function ellipsoid(rx, rz, θ_array)
    r = rx .* rz ./ sqrt.((rx .* cos.(θ_array)) .^ 2 + (rz .* sin.(θ_array)) .^ 2)
    ∂r_by_∂θ =
        (rx .^ 2 - rz .^ 2) / (rx .^ 2 * rz .^ 2) .* r .^ 3 .* sin.(θ_array) .*
        cos.(θ_array)

    n̂_r_comp = r ./ sqrt.(r .^ 2 + ∂r_by_∂θ .^ 2)
    n̂_θ_comp = -∂r_by_∂θ ./ sqrt.(r .^ 2 + ∂r_by_∂θ .^ 2)
    n̂_ϕ_comp = zeros(Real, size(r)) # TODO: I need to make the type the same as the type of r

    # n̂ = create_array_from_elements.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp) # TODO: verify that you can delete this line
    # n̂ = SVector.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)
    # n̂ = Vector.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)
    n̂ = reshape(
        [
            Vector([n̂_r_comp[id], n̂_θ_comp[id], n̂_ϕ_comp[id]]) for
            id in eachindex(n̂_r_comp)
        ],
        size(n̂_r_comp),
    )
    return r, n̂
end

"""
    Create 1D mesh, then get r, θ and n̂_θ_comp for points between each two consecutive points defined by r_edges_array, θ_edges_array

It returns a Matrix, first and second columns are for r,θ, respectively. The third, fourth and fifth columns are for the three compoonents of the normal direction.
"""
function create_1D_mesh_and_get_r_n̂(
    r_edges_array::AbstractVector{R},
    θ_edges_array::AbstractVector{R},
) where {R <: Real}
    z =
        find_point_splitting_line_between_two_points_and_normal_vector_to_this_line.(
            r_edges_array[1:(end - 1)],
            θ_edges_array[1:(end - 1)],
            r_edges_array[2:end],
            θ_edges_array[2:end],
        )
    """
    r_array = [z[id][1] for id = 1 : length(z)]
    θ_array = [z[id][2] for id = 1 : length(z)]
    n̂_array = [z[id][3] for id = 1 : length(z)]
    """
    return vcat(z...)
end

function create_1D_mesh_and_get_r_n̂_with_plotting(
    r_edges_array::AbstractVector{R},
    θ_edges_array::AbstractVector{R},
) where {R <: Real}
    z = create_1D_mesh_and_get_r_n̂(r_edges_array, θ_edges_array)
end

"""
    Find a point (r,θ) that split a line connecting two points (r1, θ1), and (r2, θ2) into two equal splits, and find the normal unit vector is spherical coordinates
"""
function find_point_splitting_line_between_two_points_and_normal_vector_to_this_line(
    r1::R,
    θ1::R,
    r2::R,
    θ2::R,
) where {R <: Real}
    S = calculate_distance_between_two_points(r1, θ1, r2, θ2)
        t=r2 * sin(θ2 - θ1) / S
        if t ≈ 1
        angle_between_r1_and_S=asin(1)
        else
        angle_between_r1_and_S = asin(t)
    end
    # to fix the inverse sin of obtuse angle
    if r2 > r1
        angle_between_r1_and_S = pi - angle_between_r1_and_S
    end
    r = sqrt(r1^2 + (S / 2)^2 - 2 * r1 * (S / 2) * cos(angle_between_r1_and_S))
    angle_between_r1_and_r = asin(S / 2 * sin(angle_between_r1_and_S) / r) # TODO: do the check for the "obtuse" angle
    θ = angle_between_r1_and_r + θ1

    γ = pi - angle_between_r1_and_S - angle_between_r1_and_r

    n̂_θ_comp = -cos(γ)
    n̂_r_comp = sin(γ)

    # println("r1=$r1, θ1=$θ1, r2=$r2, θ2=$θ2, r=$r, S=$S, θ=$θ, angle_between_r1_and_S/pi=$(angle_between_r1_and_S/pi), asin($(r2 * sin(θ2-θ1) / S)), θ2-θ1=$(θ2-θ1)")
    return hcat(r, θ, n̂_r_comp, n̂_θ_comp, zero(R))
    # return hcat(r, θ, n̂_θ_comp)
end

"""
    Find distance between two points (r1, θ1), and (r2, θ2)
"""
function calculate_distance_between_two_points(r1::R, θ1::R, r2::R, θ2::R) where {R <: Real}
    return sqrt(r1^2 + r2^2 - 2 * r1 * r2 * cos(θ2 - θ1))
end

"""
    Calculate the total length of a 1D mesh, defined as r and θ arrays.
"""
function calculate_total_length_of_1D_mesh(
    r_array::AbstractVector{R},
    θ_array::AbstractVector{R},
) where {R <: Real}
    return sum(
        calculate_distance_between_two_points.(
            r_array[1:(end - 1)],
            θ_array[1:(end - 1)],
            r_array[2:end],
            θ_array[2:end],
        ),
    )
end

"""
    Calculate the surface area of an axisymmetric particle, defined by a 1D mesh.
"""
function calculate_surface_area_of_axisymmetric_particle(
    r_array::AbstractVector{R},
    θ_array::AbstractVector{R},
) where {R <: Real}
    return sum(
        calculate_lateral_surface_area_of_truncated_cone.(
            r_array[1:(end - 1)],
            θ_array[1:(end - 1)],
            r_array[2:end],
            θ_array[2:end],
        ),
    )
end

"""
    Calculate surface area of a truncated cone defined by two points revolving about the Z axis.
"""
function calculate_lateral_surface_area_of_truncated_cone(
    r1::R,
    θ1::R,
    r2::R,
    θ2::R,
) where {R <: Real}
    r = r1 * sin(θ1)
    RR = r2 * sin(θ2)
    s = abs(r1 * cos(θ1) - r2 * cos(θ2))
    return pi * (r + RR) * sqrt((r - RR)^2 + s^2)
end

"""
    Calculate the volume of an axisymmetric particle, defined by a 1D mesh.
"""
function calculate_volume_of_axisymmetric_particle(
    r_array::AbstractVector{R},
    θ_array::AbstractVector{R},
) where {R <: Real}
    return sum(
        calculate_volume_of_truncated_cone.(
            r_array[1:(end - 1)],
            θ_array[1:(end - 1)],
            r_array[2:end],
            θ_array[2:end],
        ),
    )
end

"""
    Calculate volume of a truncated cone defined by two points revolving about the Z axis.
"""
function calculate_volume_of_truncated_cone(r1::R, θ1::R, r2::R, θ2::R) where {R <: Real}
    r = r1 * sin(θ1)
    RR = r2 * sin(θ2)
    s = abs(r1 * cos(θ1) - r2 * cos(θ2))
    return 1 / 3 * pi * (r^2 + r * RR + RR^2) * s
end

"""
    Calculate surface area for an ellipsoid
    following equations in SMARTIES paper: 1. Somerville, W. R. C., Auguié, B. & Le Ru, E. C. Smarties: User-friendly codes for fast and accurate calculations of light scattering by spheroids. J. Quant. Spectrosc. Radiat. Transf. 174, 39–55 (2016).
    verified against results from this area and volume calculator https://planetcalc.com/149/
"""
function get_volume_area_for_ellipsoid(rx, rz)
    h = rx / rz
    if h < 1
        h = 1 / h
    end  # h is the aspect ratio that is larger than 1
    e = sqrt(h^2 - 1) / h

    volume = 4 / 3 * pi * rx^2 * rz  # eq. 9
    if rx > rz  # oblate
        area_surface = (2 * pi * rx^2 * (1 + ((1 - e^2) / e) * atanh(e)))  # eq. 11
    elseif rx == rz  # sphere
        area_surface = 4 * pi * rx^2
    else  # prolate
        area_surface = 2 * pi * rx^2 * (1 + rz / (rx * e) * asin(e))  # eq. 11
    end
    return volume, area_surface
end

function get_volume_area_for_geometry(geometry_name, geometry_parameters...)
    if geometry_name in ["spheroid", "ellipsoid"]
        return get_volume_area_for_ellipsoid(geometry_parameters...)
    end
end

function convert_coordinates_Cart2Sph(x, y, z)
    r = sqrt((x^2) + (y^2) + (z^2))
    theta = atan(sqrt((x^2) + (y^2)), z)
    phi = atan(y, x)
    return hcat(r, theta, phi)
end

function convert_coordinates_Sph2Cart(r, theta, phi)
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    return hcat(x, y, z)
end

"""
    Given r and θ arrays, this function returns another r and θ arrays covering the same range of θ, with normal direction vector

Note that the first and last elements of θ_array are assumed to be 0 and pi, respectively (or numbers very close to these). We assume that the n̂ is pointing at +ve and -ve z directions at these points.
"""
function get_r_θ_n̂_arrays_from_r_θ_arrays_axisymmetrix(
    r_array::AbstractVecOrMat{R},
    θ_array::AbstractVecOrMat{R},
) where {R <: Real}
    # TODO: I need to make sure that the first and last elements of θ_array are 0 and pi, respectively (or numbers very close to these). Note that we assume that the n̂ is pointing at +ve and -ve z directions at these points.
    r_theta_n̂ = create_1D_mesh_and_get_r_n̂(r_array, θ_array)
    return vcat(
        [r_array[1], θ_array[1], one(R), zero(R), zero(R)]',
        r_theta_n̂,
        [r_array[end], θ_array[end], one(R), zero(R), zero(R)]',
    )
end

"""
Increase mesh density by 4x, by dividing every line element into two equal parts.
Works only for 1D mesh.
"""
function double_mesh_density(r_array, θ_array)
    z = Tmatrix.create_1D_mesh_and_get_r_n̂(r_array, θ_array)
    r_middles = z[:, 1]
    θ_middles = z[:, 2]
    # return hcat(r_middles, θ_middles)

    """
    r_all = vcat(r_array,r_middles)
    r_all[1:2:end] = r_array
    r_all[2:2:end] = r_middles

    θ_all = vcat(θ_array,θ_middles)
    θ_all[1:2:end] = θ_array
    θ_all[2:2:end] = θ_middles
    return hcat(r_all, θ_all)
    """

    return hcat(
        add_elements_from_two_arrays_in_turn(r_array, r_middles),
        add_elements_from_two_arrays_in_turn(θ_array, θ_middles),
    )
end

"""
Increase mesh density by 4x, by dividing every line element into four equal parts.
Works only for 1D mesh.
"""
function quadruple_mesh_density(r_array, θ_array)
    r_theta_array_double = double_mesh_density(r_array, θ_array)
    return double_mesh_density(r_theta_array_double[:, 1], r_theta_array_double[:, 2])
end

"""
Increase mesh density by 8x, by dividing every line element into eight equal parts.
Works only for 1D mesh.
"""
function octuple_mesh_density(r_array, θ_array)
    r_theta_array_quadruple = quadruple_mesh_density(r_array, θ_array)
    return double_mesh_density(r_theta_array_quadruple[:, 1], r_theta_array_quadruple[:, 2])
end

"""
Increase mesh density by 16x, by dividing every line element into sixteen equal parts.
Works only for 1D mesh.
"""
function sexdecuple_mesh_density(r_array, θ_array)
    r_theta_array_octuple = octuple_mesh_density(r_array, θ_array)
    return double_mesh_density(r_theta_array_octuple[:, 1], r_theta_array_octuple[:, 2])
end

"""
julia> a = [1,2,3]; b = [10,20];
julia> add_elements_from_two_arrays_in_turn(a,b)
julia> 5-element Vector{Int64}:
1
10
2
20
3
"""
function add_elements_from_two_arrays_in_turn(array_1, array_2)
    return array_1
    return [
        isodd(i) ? array_1[Int((i + 1) / 2)] : array_2[Int(i / 2)] for
        i in 1:(length(array_1) + length(array_2))
    ]
end
