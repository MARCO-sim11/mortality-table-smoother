# Spline cubique naturelle (zone I)

> **En une phrase.** Le logarithme de la force de mortalité est une courbe formée de morceaux de cubiques raccordés en des nœuds espacés de 5 ans ; ce sont les valeurs en ces nœuds qui sont estimées par moindres carrés pondérés.

| | |
|---|---|
| **Zone** | I : de l'âge 0 à $\omega_1$ |
| **Modèle** | $\ln\mu_x = S(x)$, $S$ spline cubique naturelle de nœuds $\tau_1 < \dots < \tau_K$ |
| **Estimation** | moindres carrés pondérés (poids $D_x$) sur la base de la spline, forme fermée |
| **Paramètres** | $K$, le nombre de nœuds (5 sur 0-20 ans) |
| **Itérations** | aucune (`n_iter = 0`) |
| **Code** | `src/zone1_poly.py`, fonction `fit_spline` |

## 1. Pourquoi une spline

Le polynôme du degré 3 impose une forme globale : une seule bosse, ou un seul creux. La spline, elle, est **locale** : chaque morceau s'adapte à son voisinage sans contraindre le reste. Elle peut donc suivre une enfance non monotone avec plus de liberté, au prix de plus de paramètres.

## 2. Le modèle

Une **spline cubique naturelle** est une fonction $S$ qui, entre deux nœuds consécutifs, est un polynôme de degré 3, avec :

- $S$, $S'$ et $S''$ continues en chaque nœud (aucun coude visible) ;
- une courbure nulle aux deux extrémités : $S''(\tau_1) = S''(\tau_K) = 0$.

Avec $K$ nœuds, l'espace de ces fonctions est de dimension $K$ : une spline naturelle est entièrement déterminée par ses **valeurs aux nœuds**, $\beta_k = S(\tau_k)$. D'où le nombre de paramètres, $K$.

Hors de l'intervalle des nœuds, la spline naturelle se prolonge par la **droite tangente** à l'extrémité : en logarithme de $\mu$, la mortalité prolongée croît ou décroît exponentiellement, sans osciller ni exploser comme le ferait la dernière cubique.

## 3. L'estimation pas à pas

**Étape 1 : les nœuds.** Un nœud tous les 5 ans à partir du premier âge de la zone (0, 5, 10, 15, 20 pour $\omega_1 = 20$). Le **dernier âge** de la zone est toujours candidat, car le raccord a besoin de $\mu$ en $\omega_1$ : il remplace le dernier nœud de la grille s'il en est à moins d'un demi-pas, sinon il est ajouté (une zone de 0 à 11 ans donne les nœuds 0, 5, 11 ; une zone de 0 à 13 ans donne 0, 5, 10, 13). Un âge candidat **sans décès est écarté**, il n'est pas déplacé.

**Étape 2 : la base.** Pour chaque nœud $\tau_k$, la fonction de base $\phi_k$ est la spline naturelle valant 1 en $\tau_k$ et 0 aux autres nœuds (base « cardinale »). Elle est construite avec `scipy.interpolate.CubicSpline(..., bc_type='natural')`. La matrice du modèle est $A_{xk} = \phi_k(x)$.

**Étape 3 : la régression.** Comme pour le polynôme, avec $y_x = \ln(D_x/E_x)$ et les poids $D_x$ :

$$\hat\beta \;=\; (A^\top W A)^{-1} A^\top W\,y,$$

résolu par `numpy.linalg.solve`. La courbe est $\hat S(x) = \sum_k \hat\beta_k\,\phi_k(x)$ et $\hat\mu_x = e^{\hat S(x)}$.

**Étape 4 : la log-vraisemblance** est calculée a posteriori, avec la formule de Poisson commune (les âges sans décès y contribuent).

### Interpolation ou régression ?

Le cahier des charges demande `CubicSpline` avec un nœud tous les 5 ans ; la note méthodologique précise que l'estimation reste pondérée, « avec la même structure, $A$ étant la base de splines ». Les deux sont réunis ici. **Interpoler** les $\ln m_x$ bruts aux nœuds recopierait dans la courbe le bruit de chaque âge isolé ; avec la **régression**, la courbe est tirée par les âges à beaucoup de décès (les nourrissons) et fait la moyenne des âges bruités. Quand chaque nœud est aussi un âge observé et qu'il n'y a pas d'autre âge, les deux coïncident.

## 4. Détails et décisions d'implémentation

- **Minimum de 3 nœuds** : avec moins, la spline naturelle n'est qu'une droite ; une erreur en français est levée (« zone I trop courte, ou trop d'âges sans décès »).
- **Âges sans décès** : hors de l'ajustement et hors des nœuds, mais $\hat\mu_x$ est évalué en **chaque** âge reçu.
- **Prolongement** : droite tangente en logarithme au-delà des nœuds extrêmes (une zone dont le dernier âge n'a aucun décès perd ce nœud).
- **Réglages** : aucun ; la règle des nœuds est fixée par le cahier des charges. L'argument `knot_step` de la fonction (5 par défaut) existe pour les expériences et les tests, pas pour l'application.
- **Paramètres rendus** : `knots` (les âges), `ln_mu_knots` (les $\hat\beta_k$) et `n_params = K`.

## 5. Forces et limites

**Forces.** Souple et locale ; forme fermée, donc sans échec numérique ; la courbe reste positive ; la mortalité d'un âge n'est pas tirée par un âge lointain.

**Limites.**

- Plus de paramètres que le polynôme (5 contre 4 sur 0-20 ans) : l'AIC la pénalise, et sur les données nationales elle n'est pas gagnante en zone I (voir le tableau).
- Avec un nœud tous les 5 ans, elle ne suit pas non plus la chute entre 0 et 1 an : le taux de l'âge 1 est surestimé d'un facteur 4 à 6 sur la France 2020.
- Plus il y a de nœuds, plus elle épouse le bruit ; moins il y en a, plus elle se rapproche d'une droite.

## 6. Résultats sur la France 2020

Réglages par défaut (zone I : 0 à 20 ans, nœuds 0, 5, 10, 15, 20).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | 5 nœuds : 0, 5, 10, 15, 20 | 21 | -223.4 | 456.9 | 12.8 | 0.133 | 33.3 % |
| Masculin | 5 nœuds : 0, 5, 10, 15, 20 | 21 | -335.3 | 680.7 | 52.6 | 0.014 | 61.9 % |
<!-- resultats:fin -->

## Références

- de Boor, C. (2001). *A Practical Guide to Splines*, éd. révisée. Springer.
- Green, P.J., Silverman, B.W. (1994). *Nonparametric Regression and Generalized Linear Models*. Chapman & Hall. (Splines naturelles.)
