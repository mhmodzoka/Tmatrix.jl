# Contains all functions handling geometry

"""
    Calculate spherical coordinates arrays representing the surface of the particle
This can be either 1D arrays for axi-symmetric particles or 2D arrays for general 3D particles
"""
function get_r_θ_ϕ_arrays(geometry; angular_resolution=0.5)
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
    
    n̂ = create_array_from_elements.(n̂_r_comp, n̂_θ_comp, n̂_ϕ_comp)
    """
    # TODO @Alok, is there an efficient way to do it?
    # this one is maybe 10x slower than the one in "else"
    n̂ = (_ -> zero(Vector{3,Real})).(r)
    for idx in CartesianIndices(n̂)
        n̂[idx] = [
            n̂_r_comp[idx],
            n̂_θ_comp[idx],
            n̂_ϕ_comp[idx],
        ]
    end
    """
    
    
    return r, n̂
end