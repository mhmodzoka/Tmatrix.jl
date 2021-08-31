import jax.numpy as np

"""
Calculate Bessel functions using recurrence relation
"""
def recurrence_relation_bessel(function_here, n: int, x): #where NN <: Number
    return (2*(n-1)/x) * function_here(n-1, x) - function_here(n-2, x)


"""
    Recurrence formula for ALL spherical Bessel functions.
I mean by spherical Bessel functions the following:
jₙ, spherical Bessel of the first kind
yₙ, spherical Bessel of the second kind
hₙ⁽¹⁾
hₙ⁽²⁾
http://dlmf.nist.gov/10.51.E1
"""
def recurrence_relation_spherical_bessel(function_here, n: int, x): #where NN <: Number
    print("hello")
    return ((2*(n-1)+1)/x) * function_here(n-1, x) - function_here(n-2, x)



def recurrence_relation_spherical_bessel(n: int, x, bessel_n_minus_1, bessel_n_minus_2): #where NN <: Number
    return (2*(n-1)+1)/x * bessel_n_minus_1 - bessel_n_minus_2


"""
    Calculate spherical Bessel function, given the Bessel functions at n=0 and n=1
Works for ALL spherical Bessel functions
"""
def spherical_Bessel_n(n: int, x, bessel_0, bessel_1): #where NN <: Number
    if n == 0:
        return bessel_0
    elif n == 1:
        return bessel_1
    else:
        # use the recurrence
        n_now = 2
        bessel_n_minus_2 = bessel_0
        bessel_n_minus_1 = bessel_1
        while n_now <= n:
            bessel_n_now = recurrence_relation_spherical_bessel(n_now, x, bessel_n_minus_1, bessel_n_minus_2)
            if n_now == n:
                return bessel_n_now
            else:
                bessel_n_minus_2 = bessel_n_minus_1
                bessel_n_minus_1 = bessel_n_now
                n_now += 1
            
        
    


# jₙ, spherical Bessel of the first kind =======================================
"""
    Spherical Bessel def of the first kind, at n=0
from http://dlmf.nist.gov/10.49.E3
"""
def spherical_Bessel_j_0(x): #where NN <: Number
    return np.sin(x)/x


"""
    Spherical Bessel def of the first kind, at n=1
from http://dlmf.nist.gov/10.49.E3
"""
def spherical_Bessel_j_1(x): #where NN <: Number
    return np.sin(x)/(x^2) - np.cos(x)/x


"""
    Spherical Bessel def of the first kind, using recurrence relation
"""
def spherical_Bessel_j_n(n: int, x): #where NN <: Number
    if n == 0:
        return spherical_Bessel_j_0(x)
    elif n == 1:
        return spherical_Bessel_j_1(x)
    else:
        bessel_0 = spherical_Bessel_j_0(x)
        bessel_1 = spherical_Bessel_j_1(x)
        return spherical_Bessel_n(n, x, bessel_0, bessel_1)
    


# yₙ, spherical Bessel of the second kind =======================================
"""
    Spherical Bessel def of the second kind, at n=0
from http://dlmf.nist.gov/10.49.E5
"""
def spherical_Bessel_y_0(x): #where NN <: Number
    return -np.cos(x)/x


"""
    Spherical Bessel def of the second kind, at n=1
from http://dlmf.nist.gov/10.49.E5
"""
def spherical_Bessel_y_1(x): #where NN <: Number
    return -np.cos(x)/(x^2) - np.sin(x)/x


"""
    Spherical Bessel def of the second kind, using recurrence relation
"""
def spherical_Bessel_y_n(n: int, x): #where NN <: Number
    if n == 0:
        return spherical_Bessel_y_0(x)
    elif n == 1:
        return spherical_Bessel_y_1(x)
    else:
        bessel_0 = spherical_Bessel_y_0(x)
        bessel_1 = spherical_Bessel_y_1(x)
        return spherical_Bessel_n(n, x, bessel_0, bessel_1)
    
