import numpy as np
from numpy.linalg import eigh,norm

def svd(macierz):
    ev, macierzV =eigh(macierz.T@macierz)
    print("Macierz V:\n" ,macierzV)
    u0= macierz@macierzV[:,0]/norm(macierz@macierzV[:,0])
    u1= macierz@macierzV[:,0]/norm(macierz@macierzV[:,0])
    u2= macierz@macierzV[:,0]/norm(macierz@macierzV[:,0])
    macierzU=np.array([u0,u1,u2]).T
    print("Macierz U:\n" ,macierzU)

a=np.array([[1,1,3],
    [5,2,4]])

svd(a)