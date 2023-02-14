import random as ran
import math

# Module for computing paramters for bifurcations
# Math derived from the article in the link
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2229505/

# In the following:
# 0 is parent artery
# 1 is biggest child artery
# 2 is smallest child artery

# Ratio of diameters between child arteries, between 0 and 1.
def getRandomAlpha():
    alpha = ran.random()
    return alpha


# Child and parent arteries diameter ratios
def getLambda(numerator,alpha):
    return numerator/(1+alpha**3)**(1/3)

def getLambdas(alpha):
    lambda1 = getLambda(1,alpha)
    lambda2 = getLambda(alpha,alpha)
    return lambda1, lambda2


# Child and parent arteries length ratios
def getGamma(alpha):
    getLambda(alpha)

def getGammas(alpha):
    Gammas = getLambdas(alpha)
    return Gammas


# Child artery angles from parent
def getTheta1(alpha):
    numerator = (1+alpha**3)**(4/3)+1-alpha**4
    denumenator = 2*(1+alpha**3)**(2/3)
    cosTheta = numerator/denumenator
    if cosTheta > 1:
        cosTheta = 1
    if cosTheta < -1:
        cosTheta = 1
    return math.acos(cosTheta)

def getTheta2(alpha):
    numerator = (1+alpha**3)**(4/3)+alpha**4-1
    denumenator = 2*alpha**2*(1+alpha**3)**(2/3)
    cosTheta = numerator/denumenator
    if cosTheta > 1:
        cosTheta = 1
    if cosTheta < -1:
        cosTheta = 1
    return math.acos(cosTheta)

# bifurcation angels
def getThetas(alpha):
    return getTheta1(alpha), getTheta2(alpha)


# gets All parameters
def getAllParameters(alpha):
    return getLambdas(alpha), getThetas(alpha)