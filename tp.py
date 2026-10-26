
# TP- Approximation de Taylor avec autograd et Streamlit 

# 1. Découverte de Streamlit
# 1.1 Première application

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

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

FONCTION={"sinus":np.sin,"cos":np.cos,"exp":np.exp,"ln(1+x)":lambda x:np.log(1+x),"custom":custom}

# 4.3 Widgets dans la barre latérale 

with st.sidebar:
    nom_fonction = st.selectbox(
        "Choisir une fonction",
        list(FONCTION.keys()))

    n = st.slider(
        "Ordre du développement de Taylor",
        min_value=0,
        max_value=10,
        value=3)

    largeur = st.slider(
        "Largeur de l'intervalle étudié",
        min_value=2.0,
        max_value=20.0,
        value=12.6,
        step=0.1)

# 4.4. calculs et tracé 

# D'abord on récupére la fonction choisie dans le dictionnaire
f = FONCTION[nom_fonction]

# On construie l'intervalle de 400 points 
X = np.linspace(-largeur / 2, largeur / 2, 400)

# On calcule les valeurs de la fonction et de son approximation
# np.errstate permet d'éviter les problèmes
# avec le ln(1+x) lorsque x ≤ -1 car il ne rend pas
#  ces valeurs définies
with np.errstate(divide="ignore", invalid="ignore"):
    Y = f(X)

Y_taylor = taylor2(X, f, n)

# Tracer les deux courbes
fig, ax = plt.subplots()

ax.plot(X, Y, label=nom_fonction)
# on peut utiliser le f" pour préciser à quel ordre on le fait
ax.plot(X, Y_taylor,label=f"Taylor à l'ordre {n}")

ax.legend()
st.pyplot(fig)

# A n fixé, l’approximation est meilleure près de 0 
# et se dégrade pas mal lorsqu’on s’éloigne. 
# Pour ln(1+x), la série de Taylor en 0 ne converge pas 
# pour x > 1, augmenter n ne permet 
# donc pas de converger vers la fonction. 
# et pour x ≤ -1, la fonction réelle n’est pas définie.


# 5. Qualité de l'approximation
# 5.1: Taux de coïncidence 

def taux_coincidence(Y, Y_taylor, seuil):
    erreur = np.abs(Y - Y_taylor)
    
    # erreur < seuil crée un tableau de booléens avec 
    # True si l’erreur est inférieure au seuil et
    # False sinon
    proches = erreur < seuil

    # On prend la moyenne pour avoir le taux 
    return np.mean(proches)


# On ajoute le choix du seuil dans la barre latérale
with st.sidebar:
    seuil = st.select_slider(
        "Seuil de précision",
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-3)


# On affiche le résultat dans la zone principale
taux = taux_coincidence(Y, Y_taylor, seuil)

st.write(f"Taux de coïncidence : {taux * 100:.2f} %")

# 5.2. Courbe d'erreur 

# 5.2 Courbe d'erreur

afficher_erreur = st.checkbox("Afficher l'erreur")

# On utilise un if 
if afficher_erreur:
    erreur = np.abs(Y - Y_taylor)

    # Pour Une échelle log, on a besoin de 
    # valeurs positives
    # On masque aussi les valeurs non définies ou infinies
    # np.where(condition, valeur_si_vrai, valeur_si_faux)

    erreur_trace = np.where(
        np.isfinite(erreur) & (erreur > 0),
        erreur,
        np.nan)

    fig_erreur, ax_erreur = plt.subplots()

    ax_erreur.plot(X, erreur_trace, label="Erreur absolue")

    # c'est là qu'on met l'axe vertical en échelle log
    ax_erreur.set_yscale("log")

    ax_erreur.set_xlabel("x")
    ax_erreur.set_ylabel("Erreur absolue")
    ax_erreur.set_title("Erreur de l'approximation de Taylor")
    ax_erreur.legend()


    st.pyplot(fig_erreur)

# 6. Comparer plusieurs ordres

st.header("Comparer plusieurs ordres")

# On créée la liste une seule fois
if "ordres" not in st.session_state:
    st.session_state.ordres = []

# On ajouter l'ordre actuellement choisi et sans doublon
# Un clic alors que l'ordre est déjà là ne change rien
if st.button("Ajouter cet ordre à la comparaison"):
    if n not in st.session_state.ordres:
        st.session_state.ordres.append(n)

# Pour réinitialiser et vider la liste
if st.button("Réinitialiser la comparaison"):
    st.session_state.ordres = []

# On afficher les ordres mémorisés
st.write("Ordres sélectionnés :", st.session_state.ordres)

# et on trace les courbes si au moins 
# un ordre a été ajouté
if st.session_state.ordres:
    fig_comparaison, ax_comparaison = plt.subplots()

    # pour la fonction originale
    ax_comparaison.plot(
        X, Y,
        label=nom_fonction)

    # et les approximations pour chaque ordre mémorisé
    for k in sorted(st.session_state.ordres):
        Y_k = taylor2(X, f, k)

        ax_comparaison.plot(
            X, Y_k,
            label=f"Taylor à l'ordre {k}")

    ax_comparaison.set_xlabel("x")
    ax_comparaison.set_ylabel("y")
    ax_comparaison.set_title("Comparaison des ordres de Taylor")
    ax_comparaison.legend()
  
    st.pyplot(fig_comparaison)

else:
    st.write("Ajoutez un ordre pour commencer la comparaison.")
    