# Perks (zone II)

> **En une phrase.** Une forme logistique de la force de mortalité, $\mu_x = (A + B\,c^x)/(1 + D\,c^x)$ : identique à Makeham tant que $D\,c^x \ll 1$, puis infléchie vers un plateau $B/D$ ; c'est le modèle de la zone II qui peut **décélérer**.

| | |
|---|---|
| **Zone** | II : de $\omega_1$ à $\omega_2$ |
| **Modèle** | $\mu_x = \dfrac{A + B\,c^x}{1 + D\,c^x}$, avec $A \ge 0$, $B > 0$, $D \ge 0$, $c > 1$ |
| **Estimation** | maximum de vraisemblance de Poisson, Nelder-Mead sur la demi-déviance |
| **Paramètres** | 4 (`n_params = 4`) |
| **Contient** | Makeham ($D = 0$) |
| **Code** | `src/zone2_perks.py`, fonction `fit_perks` |

## 1. La loi

Perks (1932) remplace la croissance exponentielle sans fin par une courbe **logistique** :

$$\mu_x \;=\; \frac{A + B\,c^x}{1 + D\,c^x}.$$

- Quand $D\,c^x \ll 1$, le dénominateur vaut 1 : on retrouve Makeham, $\mu_x \approx A + B\,c^x$.
- Quand $x$ grandit, $D\,c^x$ rattrape 1 et la mortalité est tirée vers le **plateau** $\mu_\infty = B/D$ au lieu d'exploser.

Avec $D = 0$, Perks est Makeham ; sa log-vraisemblance maximale n'est donc jamais inférieure. Comme $D \ge 0$, Perks ne sait que **ralentir** la croissance : si les données **accélèrent** au-delà de Makeham, l'optimum se trouve sur le mur $D = 0$ et l'ajustement est celui de Makeham.

## 2. L'estimation

Le cahier des charges prescrit le simplexe de Nelder-Mead : avec une forme fractionnaire à quatre paramètres, la hessienne de Newton est mal conditionnée.

### 2.1 Les étapes

1. **Départ** : l'ajustement de Makeham $(A, B, c)$ des mêmes âges, et $D$ tel que $D\,c^x$ vaille **10 %** à l'âge le plus élevé : Perks démarre à 10 % de Makeham à cet âge.
2. **Simplexe initial** : le départ et, pour chaque paramètre, le départ décalé d'environ 10 %.
3. **Nelder-Mead** sur la demi-déviance, relancé depuis le meilleur point tant qu'il améliore l'objectif de plus de `nr_tol` (au plus 5 relances).
4. **Retour aux paramètres** $(A, B, c, D)$ ; $A$, puis $D$, sont rapportés **exactement égaux à 0** quand le modèle contraint à 0 (autres paramètres ré-ajustés) est à moins de `nr_tol` du modèle libre.

### 2.2 Les coordonnées de l'optimiseur

Elles sont sans contrainte et d'ordre 1, avec l'âge centré sur l'âge moyen des décès $\bar x$ ; on pose $b = B\,c^{\bar x}$ et $\delta = D\,c^{\bar x}$ :

$$A = s_A\,\theta_A^2 \;(s_A = \text{taux brut global}),\quad b = e^{\theta_b},\quad c = e^{e^{\theta_c}},\quad \delta = \frac{\theta_D^2}{w_{\max}},\qquad w = c^{x-\bar x}.$$

Le centrage supprime la corrélation extrême entre $\ln B$ et $\ln c$, qui force autrement le simplexe à ramper dans une vallée courbe. $A$ et $D$ sont des **carrés** pour pouvoir atteindre 0 : avec une exponentielle ils dériveraient vers $10^{-17}$ pendant des milliers d'itérations (la version de laboratoire en demandait 2 200, celle-ci environ 500). L'objectif est la **demi-déviance** et non $-\ell$, dont l'ordre de grandeur ($10^6$) noierait la tolérance dans le bruit d'arrondi.

Le plafond d'itérations est `fit_common.NM_MAXITER`, pas `nr_maxiter` (un plafond de pas de Newton).

## 3. Détails et cas limites

- **Minimum** : 4 âges avec exposition et des décès à 2 âges distincts, sinon erreur en français.
- `converged` : Nelder-Mead a atteint ses tolérances et une dernière relance n'a rien gagné ; `n_iter` somme ses itérations.
- **Avertissements dans `message`** : $D = 0$ (« Perks se réduit à Makeham, les données ne montrent pas de décélération ») ; $A = 0$ ; non convergence.
- **Paramètres rendus** : `A`, `B`, `c`, `D` (base des âges) et `n_params = 4`.

## 4. Forces et limites

**Forces.** C'est le seul modèle de la zone II à pouvoir ralentir : s'il y a une décélération vers 80-85 ans, il la voit.

**Limites.**

- **Sur la France 2020 la décélération n'est pas détectée entre 20 et 85 ans** : l'optimum est sur $D = 0$ pour les deux sexes et la courbe de Perks est exactement celle de Makeham (une recherche exhaustive globale le confirme). Son AIC est alors celui de Makeham **plus 2** : un paramètre inutile, pénalisé.
- La décélération réelle des âges très élevés apparaît plutôt au-delà de $\omega_2$ : c'est le rôle de la zone III (Kannisto, Beard).
- Comme Makeham 2ᵉ ordre, c'est une optimisation sans dérivée : plus lente (environ 400 à 500 itérations) et sans garantie d'optimum global.

## 5. Résultats sur la France 2020

Réglages par défaut (zone II : 20 à 85 ans). Lire en regard de la fiche de Makeham : mêmes $A$, $B$, $c$, et $D = 0$.

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | A = 0.0001574, B = 9.535e-06, c = 1.105, D = 0 | 66 | -1366.6 | 2741.2 | 1795.0 | 0.000 | 69.7 % |
| Masculin | A = 0.0002291, B = 3.59e-05, c = 1.095, D = 0 | 66 | -870.0 | 1748.0 | 624.9 | 0.000 | 56.1 % |
<!-- resultats:fin -->

## Références

- Perks, W. (1932). On some experiments in the graduation of mortality statistics. *Journal of the Institute of Actuaries*, 63.
- Nelder, J.A., Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4).
