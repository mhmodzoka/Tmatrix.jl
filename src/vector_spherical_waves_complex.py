from numpy import complex128
import VSW_utils as vswu
import jax.numpy as np

#############################################################################################
# calculate B(θ), C(θ), P(θ)
"""
    Calculation of B(θ), equation C.19, returns array
I assume each of m, n, θ is a single number
"""
def B_mn_of_θ(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}
    return np.vstack(
        0, #replacing zero(θ), if θ is a single number that should b                  # r-component
        vswu.τₘₙ(m, n, θ),      # θ-component
        1j * vswu.πₘₙ(m, n, θ)  # ϕ-component
    ) # equation C.19


"""
    Calculation of B(θ), equation C.19, returns SVector
I assume each of m, n, θ is a single number
"""
def B_mn_of_θ_SVector(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}
    """
    I assume each of m, n, θ is a single number
    """
    return SVector(
        0, #replacing zero(θ), if θ is a single number that should b                  # r-component
        vswu.τₘₙ(m, n, θ),      # θ-component
        1j * vswu.πₘₙ(m, n, θ)  # ϕ-component
    ) # equation C.19


"""
    Calculation of C(θ), equation C.20, returns array
I assume each of m, n, θ is a single number
"""
def C_mn_of_θ(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}    
    return np.vstack(
        0, #replacing zero(θ), if θ is a single number that should b                  # r-component
        1j * vswu.πₘₙ(m, n, θ), # θ-component
        -1 * vswu.τₘₙ(m, n, θ),    # ϕ-component
    ) # equation C.20


"""
    Calculation of C(θ), equation C.20, returns SVector
I assume each of m, n, θ is a single number
"""
def C_mn_of_θ_SVector(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}   
    return SVector(
        0, #replacing zero(θ), if θ is a single number that should b                  # r-component
        1j * y(m, n, θ), # θ-component
        -1 * vswu.τₘₙ(m, n, θ),    # ϕ-component
    ) # equation C.20


# TODO use partial application for M,N, fold them into the type sig of struct P
"""
    equation C.21, returns array
"""
def P_mn_of_θ(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}
    # TODO @Alok, should I replace arrays with SMatrix? what are the drawbacks?
    # TODO: replace "0" with zero(type)
    return np.vstack(
        wignerdjmn(n, 0, m, θ), # r-component
        0, #replacing zero(θ), if θ is a single number that should b                # θ-component
        0, #replacing zero(θ), if θ is a single number that should b                # ϕ-component
    ) # equation C.21


"""
    equation C.21, returns SVector
"""
def P_mn_of_θ_SVector(m: int, n: int, θ: np.float64): # where {R <: Real, I <: Integer}
    # TODO @Alok, should I replace arrays with SMatrix? what are the drawbacks?
    # TODO: replace "0" with zero(type)
    return SVector(
        wignerdjmn(n, 0, m, θ), # r-component
        0, #replacing zero(θ), if θ is a single number that should b                # θ-component
        0, #replacing zero(θ), if θ is a single number that should b                # ϕ-component
    ) # equation C.21


# calculate B(θ,ϕ), C(θ,ϕ), P(θ,ϕ), returns Array
def B_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * B_mn_of_θ(m, n, θ) * exp(1j * m * ϕ) # equation C.16


def C_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * C_mn_of_θ(m, n, θ) * exp(1j * m * ϕ) # equation C.17


def P_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * P_mn_of_θ(m, n, θ) * exp(1j * m * ϕ) # equation C.18

# calculate B(θ,ϕ), C(θ,ϕ), P(θ,ϕ), returns SVector
def B_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * B_mn_of_θ_SVector(m, n, θ) * exp(1j * m * ϕ) # equation C.16


def C_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * C_mn_of_θ_SVector(m, n, θ) * exp(1j * m * ϕ) # equation C.17


def P_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * P_mn_of_θ_SVector(m, n, θ) * exp(1j * m * ϕ) # equation C.18


# calculate (Rg)M(kr,θ,ϕ), (Rg)N(kr,θ,ϕ), returns Array
"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def M_mn_wave(m: int, n: int, kr, θ: np.float64, ϕ: np.float64, kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, _ = get_radial_function_and_special_derivative_given_kind(kind)
    return convert(R, γ_mn(m, n)) * radial_function(n, kr) * C_mn_of_θ_ϕ(m, n, θ, ϕ)



"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def N_mn_wave(m: int, n: int, kr, θ: np.float64, ϕ: np.float64, kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, radial_function_special_derivative  = get_radial_function_and_special_derivative_given_kind(kind)
    return convert(R, γ_mn(m, n)) * (
        n * (n + 1) / kr * radial_function(n, kr) * P_mn_of_θ_ϕ(m, n, θ, ϕ)
        + (radial_function_special_derivative(n, kr) * B_mn_of_θ_ϕ(m, n, θ, ϕ))
    )


# calculate (Rg)M(kr,θ,ϕ), (Rg)N(kr,θ,ϕ), returns SVector
"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def M_mn_wave_SVector(m: int, n: int, kr, θ: np.float64, ϕ: np.float64, kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, _ = get_radial_function_and_special_derivative_given_kind(kind)
    return (typeof(Complex(θ,θ)),
        convert(R, γ_mn(m, n)) * radial_function(n, kr) * C_mn_of_θ_ϕ_SVector(m, n, θ, ϕ)
    ) # make sure the output is of the same type as the input. # TODO: find a better way



"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def N_mn_wave_SVector(m: int, n: int, kr, θ: np.float64, ϕ: np.float64, kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, radial_function_special_derivative  = get_radial_function_and_special_derivative_given_kind(kind)
    return (γ_mn(m, n).astype(np.float64) * (
            n * (n + 1) / kr * radial_function(n, kr) * P_mn_of_θ_ϕ_SVector(m, n, θ, ϕ)
            + (radial_function_special_derivative(n, kr) * B_mn_of_θ_ϕ_SVector(m, n, θ, ϕ))
        )).astype(np.complex128)

