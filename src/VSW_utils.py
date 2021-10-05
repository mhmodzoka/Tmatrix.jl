import jax.numpy as np
import jax
import bessel

"""
    Dirac delta function
returns 1 if the two inputs are equal. Otherwise, return Zero
"""
def δ(x,y):
    return x == y #TODO does this work with autodiff??
    

def wignerdjmn(s: int, m: int, n: int, θ: np.float64): #where {R <: Real, I <: Integer}
    # println("s=$s, m=$m, n=$n, θ=$θ")
    if θ == 0: #replacing zero(θ), if θ is a single number that should work # TODO: make the zero the same type of θ. e.g., 0
        d = δ(m, n)
    elif θ == np.pi:
        d = (-1)^(s - n) * δ(-n, m)
    else:
        d = 0, #replacing zero(θ), if θ is a single number that should work
        
        k_min = max(0, m - n)
        k_max = min(s + m, s - n)

        if k_max >= k_min:
            # TODO: find a better way to detect when do we need to use big integers
            if (
                (max(k_max, s + m - k_max, s - n - k_max, n - m + k_max, s + abs(m), s + abs(n)) >= 21) or
                ((np.factorial(s + m) * np.factorial(s - m) * np.factorial(s + n) * np.factorial(s - n)) < 0) or
                ((np.factorial(k_max) * np.factorial(s + m - k_max) * np.factorial(s - n - k_max) * np.factorial(n - m + k_max)) < 0)
            ):
                #s = big(s); m = big(m); n = big(n)     TODO make bigfloats work       
                k_max = min(s + m, s - n)
            for k in range(k_min, k_max+1):
                d += (-1)^k * (np.cos(θ / 2)^(2*s - 2*k + m - n) * np.sin(θ / 2)^(2*k - m + n)) / (np.factorial(k) * np.factorial(s + m - k) * np.factorial(s - n - k) * np.factorial(n - m + k))
            
            d = d*np.sqrt(np.factorial(s + m) * np.factorial(s - m) * np.factorial(s + n) * np.factorial(s - n))
            return d.astype(θ)
        else: # wigner-d is zero if there is any negative factorial
            return 0, #replacing zero(θ), if θ is a single number that should work
        
    return d.astype(θ)
"""
derivative of wigner-d with resepect to θ. Adopted from eq. B.25 from Mishchenko, M.I., Travis, L.D., and Lacis, A.A. (2002). Scattering, absorption, and emission of light by small particles (Cambridge University Press).
"""
def dwignerdjmn_dtheta(s: int, m: int, n: int, θ: np.float64, numerical_derivative=False, verysmallnumber=1e-30): #where {R <: Real, I <: Integer}
    if numerical_derivative:
        return (wignerdjmn(s, m, n, θ + verysmallnumber) - wignerdjmn(s, m, n, θ - verysmallnumber)) / (verysmallnumber * 2)
    else:
        #try
        # I don't know why this is not giving the same results as the second one.
        #    return      (m - n * cos(θ)) / sin(θ) * wignerdjmn(s, m, n, θ) + sqrt((s + n) * (s - n + 1)) * wignerdjmn(s, m, n - 1, θ) # first line of equation B.25
        #catch
        #println("I am dwignerdjmn_dtheta, s=$s; m=$m; n=$n")
        if s >= n:
            return -1 * (jax.lax.stop_gradient(m - n) * np.cos(θ)) / np.sin(θ) * wignerdjmn(s, m, n, θ) - jax.lax.stop_gradient(np.sqrt((s - n) * (s + n + 1))) * wignerdjmn(s, m, n + 1, θ) # second line of equation B.25
        else: # the second part is zero. TODO: Is there a more elegent way to do it? using short circuit for example?
            return -1 * (jax.lax.stop_gradient(m - n) * np.cos(θ)) / np.sin(θ) * wignerdjmn(s, m, n, θ) # second line of equation B.25
      
    

# Looks unecessary with python?
# """
# putting non-differentiable arguments as kwargs
# """
# function wignerdjmn_ELZOUKA(θ: np.float64; s: int=0, m: int=0, n: int=0) where {R <: Real, I <: Integer}
#     return wignerdjmn_ELZOUKA(s,m,n,θ)



def πₘₙ(m: int, n: int, θ: np.float64):# where {R <: Real, I <: Integer}
    return jax.stop_grad(m) / np.sin(θ) * wignerdjmn(n, 0, m, θ)

def τₘₙ(m: int, n: int, θ: np.float64):# where {R <: Real, I <: Integer}
    return dwignerdjmn_dtheta(n, 0, m, θ)

def sqrt_factorial_n_plus_m_over_factorial_n_minus_m(m: int, n: int):
    #possibly better: list(range), then numpy.prod

    total = 1
    for i in range (2*m):
        total = total*((n+m)-i)
    return(np.sqrt(total))

    #definitely worse: mahmoud's original method

    # if n + abs(m) > 20 # Is there a better way to do it? I did it because factorial fails for input of type Int that is larger than 20
    #     m = big(m); n = big(n)
    # end 
    # return jax.lax.stop_gradient(
    #     sqrt(factorial(n + m) / factorial(n - m))
    # )

#TODO spherical Bessel

def spherical_Hankel_h1_n(n: int, x): #where {NN <: Number, I <: Integer}
    """
    Spherical Hankel function of the first kind. It can be calculated from spherical Bessel functions of the first and second kinds as in the code:
    """
    return bessel.spherical_Bessel_j_n(n, x) + 1j * bessel.spherical_Bessel_y_n(n, x)

"""return the radial def that is appropriate with the type of VSWF
"""
def get_radial_function_and_special_derivative_given_kind(kind):
    if kind in ["regular", "incoming", 1]:
        radial_function = bessel.spherical_Bessel_j_n #this gets weird, I'm leaving
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
    return (bessel.spherical_Bessel_j_n(n - 1, x) - n / x * bessel.spherical_Bessel_j_n(n, x))

def d_hankel_1_x_over_x(n: int, x): #where {NN <: Number, I <: Integer}
    """
    Derivative of (spherical Hankel of first kind * x) divided by x
    """
    return d_bessel_1_x_over_x(n, x) + 1j * d_bessel_2_x_over_x(n, x)

def d_bessel_2_x_over_x(n: int, x):  #where {NN <: Number, I <: Integer}
    """
    Derivative of (spherical Bessel of second kind * x) divided by x
    """
    return (bessel.spherical_Bessel_y_n(n - 1, x) - n / x * bessel.spherical_Bessel_y_n(n, x))
