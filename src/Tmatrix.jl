# calculate T-matrix of individual particles using Null-Field method, using equations from:
# Mishchenko, M.I., Travis, L.D., and Lacis, A.A. (2002). Scattering, absorption, and emission of light by small particles (Cambridge University Press).

module Tmatrix

export calculate_Tmatrix_for_spheroid

include("utils.jl")
include("geometry.jl")
include("tmatrix_complex.jl")
include("tmatrix_real.jl")
include("electromagnetics.jl")

println("Tmatrix.jl package has been updated on 2021/07/24")
end
