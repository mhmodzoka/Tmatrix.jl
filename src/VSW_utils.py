import jax.numpy as np

function wignerdjmn_ELZOUKA(s: int, m: int, n: int, θ: np.float64) where {R <: Real, I <: Integer}
    # println("s=$s, m=$m, n=$n, θ=$θ")
    if θ == zero(θ) # TODO: make the zero the same type of θ. e.g., zero(θ)
        d = δ(m, n);
    elseif θ == π
        d = (-1)^(s - n) * δ(-n, m);
    else
        d = zero(θ)
        
        k_min = max(0, m - n)
        k_max = min(s + m, s - n)

        if k_max >= k_min
            # TODO: find a better way to detect when do we need to use big integers
                if (
                    (max(k_max, s + m - k_max, s - n - k_max, n - m + k_max, s + abs(m), s + abs(n)) >= 21) ||
                    ((factorial(s + m) * factorial(s - m) * factorial(s + n) * factorial(s - n)) < 0) ||
                    ((factorial(k_max) * factorial(s + m - k_max) * factorial(s - n - k_max) * factorial(n - m + k_max)) < 0)
                )
                s = big(s); m = big(m); n = big(n)            
                k_max = min(s + m, s - n)
            end
            for k in k_min:k_max
                d += (-1)^k *
                        (cos(θ / 2)^(2s - 2k + m - n) * sin(θ / 2)^(2k - m + n)) /
                        (factorial(k) * factorial(s + m - k) * factorial(s - n - k) * factorial(n - m + k))
            end
            d *= sqrt(factorial(s + m) * factorial(s - m) * factorial(s + n) * factorial(s - n))
            return convert(typeof(θ), d)
        else # wigner-d is zero if there is any negative factorial
            return zero(θ)
        end
    end

    return convert(typeof(θ), d)
end

"""
putting non-differentiable arguments as kwargs
"""
function wignerdjmn_ELZOUKA(θ: np.float64; s: int=0, m: int=0, n: int=0) where {R <: Real, I <: Integer}
    return wignerdjmn_ELZOUKA(s,m,n,θ)
end


def πₘₙ(m: int, n: int, θ: np.float64):# where {R <: Real, I <: Integer}
    return jax.stop_grad(m) / sin(θ) * wignerdjmn(n, 0, m, θ)

def τₘₙ(m: int, n: int, θ: np.float64):# where {R <: Real, I <: Integer}
    return ∂wignerdjmn_by_∂θ(n, 0, m, θ)

def sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m: int, n: int):
    #possibly better: list(range), then numpy.prod

    total = 1
    for i in range (2*m):
        total = total*((n+m)-i)
    return(sqrt(total))

    #definitely worse: mahmoud's original method

    # if n + abs(m) > 20 # Is there a better way to do it? I did it because factorial fails for input of type Int that is larger than 20
    #     m = big(m); n = big(n)
    # end 
    # return Zygote.dropgrad(
    #     sqrt(factorial(n + m) / factorial(n - m))
    # )

#TODO spherical Bessel

spherical_Bessel_j_n = None #TODO
spherical_Hankel_h1_n = None #TODO

"""return the radial def that is appropriate with the type of VSWF
"""
def get_radial_function_and_special_derivative_given_kind(kind):
    if kind in ["regular", "incoming", 1]:
        radial_function = spherical_Bessel_j_n #this gets weird, I'm leaving
        radial_function_special_derivative = d_bessel_1_x_over_x
    elif kind in ["irregular", "outgoing", 2]:
        radial_function = spherical_Hankel_h1_n
        radial_function_special_derivative = d_hankel_1_x_over_x
    else:
        print(""" 'kind' has to be one of the following: ["regular", "incoming", "irregular", "outgoing"] """)
        #make a proper error throw(DomainError(""" 'kind' has to be one of the following: ["regular", "incoming", "irregular", "outgoing"] """))

    return radial_function, radial_function_special_derivative

def d_bessel_1_x_over_x(n: int, x): #where {NN <: Number, I <: Integer}
    """
    Derivative of (spherical Bessel of first kind * x) divided by x
    """
    return (spherical_Bessel_j_n(n - 1, x) - n / x * spherical_Bessel_j_n(n, x))

def d_hankel_1_x_over_x(n: int, x): #where {NN <: Number, I <: Integer}
    """
    Derivative of (spherical Hankel of first kind * x) divided by x
    """
    return d_bessel_1_x_over_x(n, x) + im * d_bessel_2_x_over_x(n, x)

def d_bessel_2_x_over_x(n: int, x):  #where {NN <: Number, I <: Integer}
    """
    Derivative of (spherical Bessel of second kind * x) divided by x
    """
    return (spherical_Bessel_y_n(n - 1, x) - n / x * spherical_Bessel_y_n(n, x))
