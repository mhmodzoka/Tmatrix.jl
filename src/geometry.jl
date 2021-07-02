# Contains all functions handling geometry

"""
    Calculate spherical coordinates arrays representing the surface of the particle
This can be either 1D arrays for axi-symmetric particles or 2D arrays for general 3D particles
"""
function get_r_θ_ϕ_arrays(geometry; angular_resolution=0.5, θ_array=nothing, ϕ_array=nothing)
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
    #return [x1,x2,x3]
    return SVector(x1,x2,x3)
end

"""
    Claculate radial coordinate of ellipsoidal particle surface, given θ points
returns r and n̂ coordinate as a function of θ_array
"""
function ellipsoid(rx, rz, θ_array)    
    r = rx .* rz ./ sqrt.((rx .* cos.(θ_array)).^2 + (rz .* sin.(θ_array)).^2)
    ∂r_by_∂θ = (rx.^2 - rz.^2) / (rx.^2 * rz.^2) .* r.^3 .* sin.(θ_array) .* cos.(θ_array)

    n̂_r_comp = r ./ sqrt.(r.^2 + ∂r_by_∂θ.^2)
    n̂_θ_comp = -∂r_by_∂θ ./ sqrt.(r.^2 + ∂r_by_∂θ.^2)
    n̂_ϕ_comp = zeros(Real, size(r)) # TODO: I need to make the type the same as the type of r
    
    # n̂ = create_array_from_elements.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp) # TODO: verify that you can delete this line
    # n̂ = SVector.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)   
    # n̂ = Vector.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)   
    n̂ = reshape([Vector([n̂_r_comp[id], n̂_θ_comp[id], n̂_ϕ_comp[id]]) for id in eachindex(n̂_r_comp)], size(n̂_r_comp))
    return r, n̂
end


"""
    Create 1D mesh, then get r, θ and n̂_θ_comp for points between each two consecutive points defined by r_edges_array, θ_edges_array
"""
function create_1D_mesh_and_get_r_n̂(r_edges_array::AbstractVector{R}, θ_edges_array::AbstractVector{R}) where R <: Real
    z = find_point_splitting_line_between_two_points_and_normal_vector_to_this_line.(r_edges_array[1:end-1], θ_edges_array[1:end-1], r_edges_array[2:end], θ_edges_array[2:end]);
    """
    r_array = [z[id][1] for id = 1 : length(z)]
    θ_array = [z[id][2] for id = 1 : length(z)]
    n̂_array = [z[id][3] for id = 1 : length(z)]
    """
    return vcat(z...)
end

function create_1D_mesh_and_get_r_n̂_with_plotting(r_edges_array::AbstractVector{R}, θ_edges_array::AbstractVector{R}) where R <: Real
    z = create_1D_mesh_and_get_r_n̂(r_edges_array, θ_edges_array);
end

"""
    Find a point (r,θ) that split a line connecting two points (r1, θ1), and (r2, θ2) into two equal splits, and find the normal unit vector is spherical coordinates
"""
function find_point_splitting_line_between_two_points_and_normal_vector_to_this_line(r1::R, θ1::R, r2::R, θ2::R) where R <: Real
    S = calculate_distance_between_two_points(r1, θ1, r2, θ2)    
    angle_between_r1_and_S = asin(r2 * sin(θ2-θ1) / S)
    # to fix the inverse sin of obtuse angle
    if r2 > r1; angle_between_r1_and_S = pi - angle_between_r1_and_S; end
    r = sqrt(r1^2 + (S/2)^2 - 2*r1*(S/2)*cos(angle_between_r1_and_S))  
    angle_between_r1_and_r = asin(S/2 * sin(angle_between_r1_and_S)/r) # TODO: do the check for the "obtuse" angle
    θ = angle_between_r1_and_r + θ1   
    
    γ = pi - angle_between_r1_and_S - angle_between_r1_and_r
    
    n̂_θ_comp = -cos(γ)
    n̂_r_comp = sin(γ)

    #println("r1=$r1, θ1=$θ1, r2=$r2, θ2=$θ2, r=$r, S=$S, θ=$θ, angle_between_r1_and_S/pi=$(angle_between_r1_and_S/pi), asin($(r2 * sin(θ2-θ1) / S)), θ2-θ1=$(θ2-θ1)") 
    return hcat(r, θ, n̂_r_comp, n̂_θ_comp, zero(R))
    #return hcat(r, θ, n̂_θ_comp)
end

"""
    Find distance between two points (r1, θ1), and (r2, θ2)
"""
function calculate_distance_between_two_points(r1::R, θ1::R, r2::R, θ2::R) where R <: Real
    return sqrt(r1^2 + r2^2 - 2*r1*r2*cos(θ2-θ1))
end

function convert_coordinates_Cart2Sph(x,y,z)
    r = sqrt((x^2) + (y^2) + (z^2))
    theta = atan(sqrt((x^2) + (y^2)), z)
    phi = atan(y, x)
    return hcat(r,theta,phi)
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
function get_r_θ_n̂_arrays_from_r_θ_arrays_axisymmetrix(r_array::AbstractVecOrMat{R}, θ_array::AbstractVecOrMat{R}) where R <: Real
    # TODO: I need to make sure that the first and last elements of θ_array are 0 and pi, respectively (or numbers very close to these). Note that we assume that the n̂ is pointing at +ve and -ve z directions at these points.
    r_theta_n̂ = create_1D_mesh_and_get_r_n̂(r_array, θ_array)
    return vcat(
        [r_array[1], θ_array[1], one(R), zero(R), zero(R)]',
        r_theta_n̂,
        [r_array[end], θ_array[end], one(R), zero(R), zero(R)]',
    )
end


using Plots


"""
################ testing
rx = 1; rz=5;
θ_array_input = collect(LinRange(0, 2*pi, 23))+rand(23)*1e-1;
r_ellips, n̂_ellips = ellipsoid(rx, rz, θ_array_input);
r_theta_nthcomp = create_1D_mesh_and_get_r_n̂(r_ellips, θ_array_input);

n_vector_middles_theta_comp = r_theta_nthcomp[:,3]

xyz_ellipse_edges = vcat(convert_coordinates_Sph2Cart.(r_ellips, θ_array_input, zeros(size(r_ellips)))...);
xys_ellipse_middles = vcat(convert_coordinates_Sph2Cart.(r_theta_nthcomp[:,1], r_theta_nthcomp[:,2], zeros(size(r_theta_nthcomp[1])))...);

#n_vector_middles = hcat(ones(size(n_vector_middles_theta_comp)),n_vector_middles_theta_comp, zeros(size(n_vector_middles_theta_comp)))
n_vector_middles = r_theta_nthcomp[:,3:5]
n_vector_middles_cartesian = vcat(convert_coordinates_Sph2Cart.(n_vector_middles[:,1], n_vector_middles[:,2], n_vector_middles[:,3])...)

plot(xyz_ellipse_edges[:,1], xyz_ellipse_edges[:,3], aspect_ratio=:equal)
plot!(xys_ellipse_middles[:,1], xys_ellipse_middles[:,3], aspect_ratio=:equal)
quiver!(xys_ellipse_middles[:,1], xys_ellipse_middles[:,3], quiver=(n_vector_middles_cartesian[:,1], n_vector_middles_cartesian[:,3]), aspect_ratio=:equal)


r_theta_n̂ = get_r_θ_n̂_arrays_from_r_θ_arrays_axisymmetrix(r_ellips, θ_array_input)
"""