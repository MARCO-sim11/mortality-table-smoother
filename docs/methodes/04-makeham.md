# Makeham (zone II)

> **En une phrase.** Gompertz auquel on ajoute une constante : $\mu_x = A + B\,c^x$, où $A$ est une mortalité indépendante de l'âge ; la vraisemblance est maximisée par un **profil sur $c$** (une grille de 100 valeurs, affinée), avec à chaque $c$ un Newton-Raphson $2\times 2$ sur $(A, B)$ sous la contrainte $A \ge 0$.

| | |
|---|---|
| **Zone** | II : de $\omega_1$ à $\omega_2$ |
| **Modèle** | $\mu_x = A + B\,c^x$, avec $A \ge 0$, $B > 0$, $c > 1$ |
| **Estimation** | maximum de vraisemblance de Poisson, profil sur $c \in [1{,}05\,;\,1{,}15]$ (100 points, affiné) |
| **Paramètres** | 3 (`n_params = 3`) |
| **Contient** | Gompertz ($A = 0$) |
| **Code** | `src/zone2_makeham.py`, fonction `fit_makeham` |

## 1. La loi

Makeham (1860) ajoute à la loi de Gompertz un terme **constant** : une mortalité par cause « externe » (accidents, maladies infectieuses) qui ne dépend pas de l'âge, alors que le terme exponentiel $B\,c^x$ représente le vieillissement.

$$\mu_x \;=\; \underbrace{A}_{\text{indépendant de l'âge}} \;+\; \underbrace{B\,c^x}_{\text{vieillissement}} .$$

Avec $A = 0$ on retrouve Gompertz : la log-vraisemblance maximale de Makeham ne peut donc jamais être inférieure à celle de Gompertz sur les mêmes âges.

## 2. L'estimation

La log-vraisemblance de Poisson est

$$\ell(A, B, c) = \sum_x \Big[\, D_x\ln(A + B\,c^x) \;-\; E_x\,(A + B\,c^x)\Big] \quad(+\text{ constantes}).$$

Trois paramètres, dont un ($c$) qui entre de façon non linéaire. L'idée : à $c$ fixé, $\mu_x = A + B\,c^x$ est **linéaire en $(A, B)$** ; le sous-problème est alors concave et se résout très bien.

### 2.1 Le profil sur $c$

1. On parcourt une grille $c_j$ de **100 points** sur $[1{,}05\,;\,1{,}15]$.
2. Pour chaque $c_j$, on résout le sous-problème $(A, B)$ (voir ci-dessous).
3. On garde le $c_j$ dont la **log-vraisemblance profilée** $\ell^*(c) = \max_{A,B}\ell(A,B,c)$ est la plus grande.
4. **Raffinement** : on affine $c$ dans la cellule gagnante (recherche scalaire bornée, entre les deux voisins du meilleur point), puis on prend $(A, B)$ en ce $c$.

### 2.2 Le sous-problème $(A, B)$ à $c$ fixé

Gradient et hessienne (analytiques) :

$$\frac{\partial\ell}{\partial A} = \sum_x\Big(\frac{D_x}{\mu_x} - E_x\Big), \qquad \frac{\partial\ell}{\partial B} = \sum_x\Big(\frac{D_x}{\mu_x} - E_x\Big)c^x, \qquad H = -\sum_x \frac{D_x}{\mu_x^2}\begin{pmatrix}1 & c^x\\ c^x & c^{2x}\end{pmatrix}.$$

$\ell$ est concave en $(A, B)$. Mais la contrainte $A \ge 0$, qu'un Newton nu ignore, compte : sur ces données, une première version de laboratoire, en Newton nu, s'était installée près de la paroi $A = 0$, **2,7 unités de log-vraisemblance** sous l'optimum. Le sous-problème est donc traité en trois temps :

1. **Le point $A = 0$** : l'optimum sur cette frontière est exact, $B = \sum D_x / \sum E_x c^x$. Si quitter ce point ne peut pas augmenter la log-vraisemblance de plus de `nr_tol` (test de Kuhn-Tucker : le score en $A$ y est $\le 0$, ou le gain estimé $g^2/(2S)$ avec le complément de Schur $S$ de la courbure est sous la tolérance), c'est **l'optimum sous contrainte**, sans itération : Makeham s'est réduit à Gompertz.
2. **Le point $B = 0$** (force constante $A = \sum D / \sum E$) : même test.
3. **Sinon**, l'optimum est intérieur ($A > 0$, $B > 0$) ; Newton-Raphson $2\times 2$ avec pas réduit de moitié converge depuis un point de départ réalisable.

### 2.3 Choix d'implémentation

- **Raffinement de $c$.** Une recherche sur grille s'arrête au meilleur point de la grille. Une grille de 100 points laisse $c$ jusqu'à 0,0005 de l'optimum, ce qui coûte 0,3 à 1,5 unité de log-vraisemblance sur la France 2020 : assez pour brouiller une comparaison d'AIC. Le raffinement corrige cela (`REFINE_C = False` s'en tient au meilleur point de la grille).
- **Initialisation.** Un départ unique serait $A_0 = 0{,}001$, $B_0 = e^{\alpha_{\text{Gompertz}}}$, $c_0 = e^{\beta_{\text{Gompertz}}}$. Avec un profil, il faut un départ à chaque $c$ : $A_0$ = 10 % du taux brut global et $B_0$ qui reproduit le nombre de décès observé. Les deux sont réalisables pour tous les $c$ de la grille, ce qu'un couple $(B_0, c_0)$ unique n'est pas.
- **Coordonnées.** $B\,c^x$ est calculé comme $b\,w_x$ avec $w_x = c^{x - x_{\text{ref}}} \le 1$ et $b = B\,c^{x_{\text{ref}}}$, $x_{\text{ref}}$ étant l'âge le plus élevé : $c^x$ seul s'étale sur dix ordres de grandeur et rend la hessienne mal conditionnée. Newton est invariant par ce changement d'échelle ; $B = b/c^{x_{\text{ref}}}$ est rapporté.

## 3. Détails et cas limites

- Tous les âges avec $E_x > 0$ entrent dans la vraisemblance (âges sans décès compris) ; les autres sont ignorés mais reçoivent un taux.
- **Minimum** : 3 âges avec exposition et des décès à 2 âges distincts, sinon erreur en français.
- `converged` est vrai si **tous** les sous-problèmes $(A, B)$ ont convergé ; `n_iter` somme les pas de Newton de la grille et du raffinement (les cas de frontière n'en coûtent aucun).
- **Avertissements dans `message`** : $A = 0$ (Makeham se réduit à Gompertz) ; $B = 0$ (force constante, $c$ n'est pas identifié) ; $c$ sur une borne de $[1{,}05\,;\,1{,}15]$ (l'optimum est peut-être au-delà).
- **Paramètres rendus** : `A`, `B`, `c`, `n_params = 3` ; $A$ vaut exactement 0.0 quand la contrainte est active.

## 4. Forces et limites

**Forces.** Un terme de plus que Gompertz pour une vraie raison démographique ; la grille protège des maxima locaux en $c$ ; la gestion explicite de la frontière $A = 0$ évite le piège du Newton nu.

**Limites.**

- Comme Gompertz, elle croît exponentiellement : elle ne ralentit pas vers 85 ans.
- Le profil est lent à côté d'un Newton unique (une centaine de sous-problèmes), et $c$ est borné à $[1{,}05\,;\,1{,}15]$ : une population très atypique pourrait vouloir sortir de la grille (l'application le signale).
- Sur la France 2020, Makeham améliore très nettement Gompertz (voir le tableau) mais reste loin de Makeham 2ᵉ ordre.

## 5. Résultats sur la France 2020

Réglages par défaut (zone II : 20 à 85 ans).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | A = 0.0001574, B = 9.535e-06, c = 1.105 | 66 | -1366.6 | 2739.2 | 1793.0 | 0.000 | 69.7 % |
| Masculin | A = 0.0002291, B = 3.59e-05, c = 1.095 | 66 | -870.0 | 1746.0 | 622.9 | 0.000 | 56.1 % |
<!-- resultats:fin -->

## Références

- Makeham, W.M. (1860). On the law of mortality. *Journal of the Institute of Actuaries*, 8.
- Richards, S.J. (2012). A handbook of parametric survival models for actuarial use. *Scandinavian Actuarial Journal*.
