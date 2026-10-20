# 2. Découverte d'autograd

import autograd.numpy as np
from autograd import grad
import matplotlib as plt
import matplotlib.pyplot as plt


# 2.1 Domaine

# 1.
X = np.linspace(0, 2 * np.pi, 100)

# 2.

def sin(x):
    return np.vectorize(np.sin)(x)


def sin_der(x):
    return np.vectorize(grad(np.sin))(x)

#Y=sin_der(X)

def cos(x):
    return np.vectorize(np.cos)(x)

#Z=cos(X)

# 3. Formule de Taylor 
# 3.1 Version 1: dérivées fournies

from math import factorial

def taylor1(X, derivatives):
    Y=np.zeros(len(X))
    for i in range(len(derivatives)):
        Y=Y+(derivatives[i]/factorial(i))*(X**i)
    return Y

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

#print(taylor1(X, sinus10))

#plt.plot(X,taylor1(X,sinus10))
#plt.show()

# 3.2 Version 2: dérivées calculées avec autograd

def taylor2(X, f, n):
    # On construit l'ancien derivatives
    derivatives=[]
    g=f
    for i in range(n+1):
        derivatives.append(g(0.0))
        g=grad(g)
    return taylor1(X,derivatives)

# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
#Y = np.sin(X)
#Y2 = taylor2(X, np.sin, 10)

#plt.plot(X,Y)
#plt.plot(X,Y2)
#plt.show()

#plt.plot(X,cos(X))
#plt.plot(X,taylor2(X,cos,0))
#plt.show()

# 3.3 Une fonction personnalisée

def custom(X):
    return (2*sin(X)+cos(X/4))

#Y3=custom(X)
#Y4=taylor2(X,custom,14)
#plt.plot(X,Y3)
#plt.plot(X,Y4)
#plt.show()

# 4. L'application: widgets et graphique 













