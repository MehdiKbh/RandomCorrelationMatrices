import numpy as np
import pandas as pd
import scipy as sc
import scipy.linalg as la
import scipy.special as sp
import math as m
import matplotlib.pyplot as plt

import sys
import HighamMethod as hg



def randcorr(p):
    #Step 1 - Generate thetas
    theta = np.zeros((p,p))
    for j in range(p-1):
        theta[j+1:p,j] = generate_theta(p-(j+1), p-(j+1))
    
    #Step 2 - Build the lower triangular Cholesky factor
    B = np.zeros((p,p))
    #First column
    B[:,0] = np.cos(theta[:,0])
    #Other columns
    for j in range(1,p):
        for i in range(j,p):
            if(i==j):
                #Diagonal
                B[i,j] = np.prod(np.sin(theta[i,:j]))
            else:
                #Off-diagonal entries
                B[i,j] = m.cos(theta[i,j])*np.prod(np.sin(theta[i,:j]))
                
                
    #Compute Correlation Matrix
    R = B @ B.T
    
    #Ensure diagonal is unit
    for i in range(p): R[i,i] = 1
    
    return R
    

def generate_theta(k, n):
    
    """
        generate_theta(k,n) generates a n vector of independant random variables distributed as a sin^k distribution 
        (using an acceptance-rejection method)
        k: parameter of the distribution
        n: size of the vector 
        
        Reference: Enes Makalic and Daniel F. Schmidt,
                   An efficient algorithm for sampling from sin^k(x) for generating random correlation matrices
                   arxiv, 2018
    """
    
    x = m.pi * np.random.beta(k+1, k+1,size=n)   #Generate X - Gamma(k+1, k+1)
    u = np.random.rand(n)   #Generate U uniform(0,1)
    #Tests for accepting
    a = np.log(u)/float(k)
    b = np.log((m.pi*m.pi*np.sin(x))/(4*x*(m.pi-x)))
    accept = (np.all(a)<=np.all(b))
    while not accept:
        
        #Repeat if rejected
        x = m.pi * np.random.beta(k+1, k+1,size=n)
        u = np.random.rand(n)
        a = np.log(u)/float(k)
        b = np.log((m.pi*m.pi*np.sin(x))/(4*x*(m.pi-x)))
        accept = (np.all(a)<=np.all(b))
        
    return x