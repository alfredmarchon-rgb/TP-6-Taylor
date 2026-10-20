
# TP- Approximation de Taylor avec autograd et Streamlit 

# 1. Découverte de Streamlit
# 1.1 Première application

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as pl

def sin(x):
    return np.vectorize(np.sin)(x)

def sin_der(x):
    return np.vectorize(grad(np.sin))(x)

def cos(x):
    return np.vectorize(np.cos)(x)

def custom(X):
    return (2*sin(X)+cos(X/4))

def taylor1(X, derivatives):
    Y=np.zeros(len(X))
    for i in range(len(derivatives)):
        Y=Y+(derivatives[i]/factorial(i))*(X**i)
    return Y

def taylor2(X, f, n):
    # On construit l'ancien derivatives
    derivatives=[]
    g=f
    for i in range(n+1):
        derivatives.append(g(0.0))
        g=grad(g)
    return taylor1(X,derivatives)


st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.header("Taylor on arrive")

# 1.2 Afficher une formule

st.latex(r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!}x^k")

# 4.2 Catalogue de fonctions

FONCTION={"sinus":np.sin(),"cos":np.cos(),"exp":np.exp(),"ln(1+x)":lambda x:np.log(1+x),"custom":cutom()}

# 4.3 Widgets dans la barre latérale 




