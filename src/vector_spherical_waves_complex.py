import VSW_utils
import jax.numpy as np

# calculate B(θ,ϕ), C(θ,ϕ), P(θ,ϕ), returns Array
def B_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * B_mn_of_θ(m, n, θ) * exp(im * m * ϕ) # equation C.16


def C_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * C_mn_of_θ(m, n, θ) * exp(im * m * ϕ) # equation C.17


def P_mn_of_θ_ϕ(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * P_mn_of_θ(m, n, θ) * exp(im * m * ϕ) # equation C.18


# calculate B(θ,ϕ), C(θ,ϕ), P(θ,ϕ), returns SVector
def B_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * B_mn_of_θ_SVector(m, n, θ) * exp(im * m * ϕ) # equation C.16


def C_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * C_mn_of_θ_SVector(m, n, θ) * exp(im * m * ϕ) # equation C.17


def P_mn_of_θ_ϕ_SVector(m: int, n: int, θ: np.float64, ϕ: np.float64): # where {R <: Real, I <: Integer}
    return (-1)^m * convert(R, sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m, n)) * P_mn_of_θ_SVector(m, n, θ) * exp(im * m * ϕ) # equation C.18


# calculate (Rg)M(kr,θ,ϕ), (Rg)N(kr,θ,ϕ), returns Array
"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def M_mn_wave(m: int, n: int, kr, θ: np.float64, ϕ: np.float64; kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, _ = get_radial_function_and_special_derivative_given_kind(kind)
    return convert(R, γ_mn(m, n)) * radial_function(n, kr) * C_mn_of_θ_ϕ(m, n, θ, ϕ)



"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def N_mn_wave(m: int, n: int, kr, θ: np.float64, ϕ: np.float64; kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
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
def M_mn_wave_SVector(m: int, n: int, kr, θ: np.float64, ϕ: np.float64; kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, _ = get_radial_function_and_special_derivative_given_kind(kind)
    return convert.(typeof(Complex(θ,θ)),
        convert(R, γ_mn(m, n)) * radial_function(n, kr) * C_mn_of_θ_ϕ_SVector(m, n, θ, ϕ)
    ) # make sure the output is of the same type as the input. # TODO: find a better way



"""
    Parameters
    ==========
    kind: string, either ["regular" or "incoming"] or ["irregular" or "outgoing"]
"""
def N_mn_wave_SVector(m: int, n: int, kr, θ: np.float64, ϕ: np.float64; kind="regular"): #where {R <: Real, I <: Integer, NN <: Number}    
    radial_function, radial_function_special_derivative  = get_radial_function_and_special_derivative_given_kind(kind)
    return convert.(typeof(Complex(θ,θ)),
            convert(R, γ_mn(m, n)) * (
            n * (n + 1) / kr * radial_function(n, kr) * P_mn_of_θ_ϕ_SVector(m, n, θ, ϕ)
            + (radial_function_special_derivative(n, kr) * B_mn_of_θ_ϕ_SVector(m, n, θ, ϕ))
        )
    )

