import random as ran
import math

# Module for computing paramters for bifurcations
# Math derived from the article in the link

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2229505/
#Undersøg hvor
# In the following:
# 0 is parent artery
# 1 is biggest child artery
# 2 is smallest child artery

# Ratio of diameters between child arteries, between 0 and 1.
def get_random_alpha():
    alpha = ran.random()
    return alpha


# Child and parent arteries diameter ratios
def get_lambda(numerator,alpha):
    return numerator/(1+alpha**3)**(1/3)

def get_lambdas(alpha):
    lambda_1 = get_lambda(1,alpha)
    lambda_2 = get_lambda(alpha,alpha)
    return lambda_1, lambda_2


# Child and parent arteries length ratios
def get_gamma(alpha):
    get_lambda(alpha)

def get_gammas(alpha):
    gammas = get_lambdas(alpha)
    return gammas


# Child artery angles from parent
def get_theta_1(alpha):
    numerator = (1+alpha**3)**(4/3)+1-alpha**4
    denumenator = 2*(1+alpha**3)**(2/3)
    cos_theta = numerator/denumenator
    # ensures cos_theta is in correct domain
    if cos_theta > 1:
        cos_theta = 1
    if cos_theta < -1:
        cos_theta = 1
    return math.acos(cos_theta)

def get_theta_2(alpha):
    numerator = (1+alpha**3)**(4/3)+alpha**4-1
    denumenator = 2*alpha**2*(1+alpha**3)**(2/3)
    cos_theta = numerator/denumenator
    # ensures cos_theta is in correct domain
    if cos_theta > 1:
        cos_theta = 1
    if cos_theta < -1:
        cos_theta = 1
    return math.acos(cos_theta)

# bifurcation angels
def get_thetas(alpha):
    return get_theta_1(alpha), get_theta_2(alpha)


# gets All parameters
def get_all_parameters(alpha):
    return get_lambdas(alpha), get_thetas(alpha)