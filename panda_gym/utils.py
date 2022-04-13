import numpy as np

def distance(a, b):
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)
#iaohdoashs
def threshold_convergence(input):
    k = 5  #the enlargement scale
    # a = 0.2 #speed of convergence for step-down
    b = 2.5   #speed of convergence for exponential
    c = input
    n = 50.0 #number of success
    e = 0.04 #original size of the target
    div = c/n

    size = (1+(k-1)/(b**(div)))*(e)  #exponential convergence
    # size = (1+(k-1)(1-(c/n)))(e)    #linearly convergence
    # size = (1+(k-1)(1-a(c/n)))(e)   #step-down convergence
    
    return size

def comparee(a1,b1,a2,b2):
    penalty1 = 0
    penalty2 = 0
    if (sum(abs(a1-a2))) > 0.01:
        penalty1=5
    if (sum(abs(b1-b2))) > 0.01:
        penalty2=5

    return -(penalty1+penalty2)

def compareemoreobj(a1,b1,c1,d1,e1,a2,b2,c2,d2,e2):
    penalty1 = 0
    penalty2 = 0
    penalty3 = 0
    penalty4 = 0
    penalty5 = 0
    if (sum(abs(a1-a2))) > 0.01:
        penalty1=1
    if (sum(abs(b1-b2))) > 0.01:
        penalty2=1    
    if (sum(abs(c1-c2))) > 0.01:
        penalty3=1
    if (sum(abs(d1-d2))) > 0.01:
        penalty4=1    
    if (sum(abs(e1-e2))) > 0.01:
        penalty5=1
    return -(penalty1+penalty2+penalty3+penalty4+penalty5)