# Beard (zone III)

> **En une phrase.** Une logistique dont le **plateau est libre**, $\mu_x = \dfrac{B\,c^x}{1 + K\,B\,c^x}$ (plateau $1/K$) : Kannisto en est le cas $K = 1$ et Gompertz le cas $K = 0$ ; trois paramètres ajustés par Nelder-Mead depuis l'ajustement de Kannisto.

| | |
|---|---|
| **Zone** | III : de $\omega_2$ à $\omega_3$ |
| **Modèle** | $\mu_x = \dfrac{B\,c^x}{1 + K\,B\,c^x}$, avec $B > 0$, $c > 1$, $K \ge 0$ |
| **Estimation** | maximum de vraisemblance de Poisson, Nelder-Mead sur la demi-déviance |
| **Paramètres** | 3 (`n_params = 3`) |
| **Plateau** | $1/K$ ; $K = 0$ : pas de plateau (Gompertz) |
| **Contient** | Kannisto ($K = 1$), Gompertz ($K = 0$) |
| **Code** | `src/zone3.py`, fonction `fit_beard` |

## 1. La loi

Beard (1959) généralise la logistique de Kannisto en laissant le **niveau du plateau** aux données :

$$\mu_x \;=\; \frac{B\,c^x}{1 + K\,B\,c^x}, \qquad \mu_\infty = \frac{1}{K}.$$

- $K = 1$ : on retrouve Kannisto. En effet avec $B = a\,e^{-b\,\omega_2}$ et $c = e^{b}$ on a $B\,c^x = a\,e^{b(x-\omega_2)}$, d'où $\mu_x = a\,e^{bt}/(1 + a\,e^{bt})$.
- $K = 0$ : $\mu_x = B\,c^x$, c'est Gompertz, sans plateau.
- Entre les deux, $1/K$ est le taux maximal atteignable : un plateau **supérieur à 1** est permis ($K < 1$), ce qui compte quand les taux bruts de l'extrême vieillesse dépassent 1.

Comme Kannisto est un cas particulier, la log-vraisemblance maximale de Beard ne peut pas lui être inférieure sur les mêmes lignes.

## 2. L'estimation

### 2.1 Les étapes

1. **Départ** : l'ajustement de Kannisto, qui **est** une courbe de Beard avec $K = 1$.
2. **Nelder-Mead** sur la **demi-déviance** ($= -\ell$ à une constante près), simplexe initial d'environ 10 % autour du départ, relancé depuis le meilleur point tant qu'il améliore l'objectif.
3. **Le mur $K = 0$** : $K$ est rapporté exactement égal à 0 quand le modèle contraint à $K = 0$ (Gompertz, autres paramètres ré-ajustés) est à moins de `nr_tol` du modèle libre (`refit_with_zero`).

### 2.2 Les coordonnées de l'optimiseur

Sans contrainte et d'ordre 1, avec l'âge compté depuis $\omega_2$ : $(\ln\beta,\ \ln\gamma,\ \sqrt K)$, où $\beta = B\,c^{\omega_2}$ (le niveau de la loi à $\omega_2$) et $\gamma = \ln c$. $K$ est un **carré** pour pouvoir atteindre 0. $B = \beta\,e^{-\gamma\,\omega_2}$ et $c = e^{\gamma}$ sont rapportés.

### 2.3 Pourquoi Nelder-Mead

La note méthodologique propose un Newton-Raphson $3\times 3$ ; le cahier des charges demande Nelder-Mead, qui est utilisé. Une version de laboratoire en BFGS à gradient numérique ne signalait pas de convergence sur la France 2020 ; celle-ci, si. Les plafonds sont ceux de `fit_common` (2 000 itérations par exécution, 5 relances au plus), pas `nr_maxiter` (un plafond de pas de Newton) ; `nr_tol` est la tolérance de l'objectif. La vraisemblance est celle de Poisson, comme pour Kannisto.

## 3. Ce que couvre le résultat

Comme Kannisto : `ages` va de $\omega_2$ à $\omega_3$ et `mu_hat` donne la courbe sur tous ces âges, **extrapolée** au-delà de la dernière ligne de données.

## 4. Détails et cas limites

- **Minimum** : 3 âges avec exposition (trois paramètres), des décès à 2 âges distincts.
- **Avertissement** : $K = 0$ — « Beard se réduit à Gompertz, aucun plateau n'est détecté ».
- **Paramètres rendus** : `B`, `c`, `K` (base des âges) et `n_params = 3`. $K$ vaut exactement 0.0 quand il est sur son mur.
- Sur la France 2020 : $K \approx 0{,}86$ (femmes) et $0{,}74$ (hommes), soit des plateaux d'environ 1,17 et 1,35.

## 5. Forces et limites

**Forces.** Un plateau qui s'adapte aux données ; Kannisto et Gompertz comme cas particuliers, donc la comparaison par l'AIC dit si le troisième paramètre est utile.

**Limites.**

- **Trois paramètres** pour des âges où l'on observe peu de décès : l'ajustement est plus fragile que celui de Kannisto, et $B$ et $c$ sont fortement corrélés.
- **La règle de sélection peut l'écarter** : Beard a le meilleur AIC de la zone III pour les deux sexes, mais le test des séquences de ses résidus des femmes est significatif ($p \approx 0{,}025 < 0{,}05$, voir le tableau) ; la règle suggérée retient alors Kannisto pour les femmes et Beard pour les hommes.
- L'extrapolation de 109 à 110 ans, comme pour Kannisto, est une hypothèse de forme.

## 6. Résultats sur la France 2020

Réglages par défaut (zone III : 85 à 110 ans ; 25 âges ajustés).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | B = 1.089e-07, c = 1.169, K = 0.8586 | 25 | -302.9 | 611.8 | 0.0 | 0.025 | 32.0 % |
| Masculin | B = 4.28e-07, c = 1.156, K = 0.7425 | 25 | -127.3 | 260.5 | 0.0 | 0.336 | 12.0 % |
<!-- resultats:fin -->

## Références

- Beard, R.E. (1959). Note on some mathematical mortality models. In *CIBA Foundation Colloquia on Ageing*, 5.
- Kannisto, V. (1994). *Development of oldest-old mortality, 1950-1990*. Odense University Press.
- Nelder, J.A., Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4).
