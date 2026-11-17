# Makeham du second ordre (zone II)

> **En une phrase.** Makeham avec une **seconde exponentielle**, $\mu_x = A + B\,c^x + D\,d^x$ : cinq paramètres pour suivre la transition vers les âges les plus élevés, estimés par maximum de vraisemblance de Poisson avec Nelder-Mead, en partant de l'ajustement de Makeham.

| | |
|---|---|
| **Zone** | II : de $\omega_1$ à $\omega_2$ |
| **Modèle** | $\mu_x = A + B\,c^x + D\,d^x$, avec $A \ge 0$, $B > 0$, $c > 1$, $D \ge 0$, $d > 1$ |
| **Estimation** | maximum de vraisemblance de Poisson, Nelder-Mead sur la demi-déviance |
| **Paramètres** | 5 (`n_params = 5`) |
| **Contient** | Makeham ($D = 0$), donc Gompertz |
| **Code** | `src/zone2_makeham2.py`, fonction `fit_makeham2` |

> Attention au nom : « Makeham 2ᵉ ordre » désigne ici la loi à **deux** exponentielles, pas une « seconde loi de Makeham » au sens de la littérature. La seconde base est notée $d$ dans cette fiche et dans le code.

## 1. La loi

On ajoute à Makeham un second terme exponentiel, de base $d$ **différente** de $c$ :

$$\mu_x \;=\; A \;+\; B\,c^x \;+\; D\,d^x .$$

Deux exponentielles de bases différentes suffisent à courber le logarithme de la mortalité : la mortalité croît d'abord à la vitesse $\ln c$, puis, quand le terme le plus pentu ($d > c$) prend le dessus, à une vitesse qui s'en rapproche. Avec $D = 0$, on retrouve exactement Makeham ; la log-vraisemblance maximale ne peut donc jamais être inférieure à celle de Makeham.

**Les deux exponentielles jouent des rôles symétriques** : échanger $(B, c)$ et $(D, d)$ donne la même courbe. Le résultat est donc rapporté dans un **ordre canonique**, $c \le d$ : $(B, c)$ est le terme à croissance lente, $(D, d)$ le terme plus pentu.

**Lire $D$ avec précaution.** Comme $d^x$ est énorme (par exemple $1{,}35^{85} \approx 10^{11}$), $D$ est minuscule sans que le terme soit négligeable : sur la France 2020 (femmes), $D \approx 10^{-13}$ mais $D\,d^{85} \approx 0{,}026$, à comparer à $B\,c^{85} \approx 0{,}033$. Pour juger du poids du second terme, comparer $B\,c^x$ et $D\,d^x$ **à un âge de référence**, pas $B$ et $D$.

## 2. L'estimation

### 2.1 Pourquoi Nelder-Mead

Avec cinq paramètres, dont deux bases, la hessienne est mal conditionnée et fortement corrélée (les $\ln B$ et $\ln c$ se compensent). On utilise donc la méthode du **simplexe de Nelder-Mead**, sans dérivée.

### 2.2 Les étapes

1. **Départ** : l'ajustement de Makeham $(A, B, c)$ des mêmes âges (`fit_makeham`, avec les mêmes réglages) ; on y ajoute une seconde exponentielle de base $d = 1{,}1\,c$ dont l'amplitude vaut **10 %** du terme de Makeham à l'âge le plus élevé. Le second terme doit partir **petit et plus pentu** que le premier : un second terme grand au départ envoie le simplexe vers la copie échangée du même optimum ou vers un optimum local moins bon.
2. **Simplexe initial** : le départ et, pour chaque paramètre, le départ décalé d'environ 10 %.
3. **Nelder-Mead** sur la **demi-déviance** (équivalente à $-\ell$ à une constante près), relancé depuis le meilleur point tant que cela améliore l'objectif de plus de `nr_tol`.
4. **Retour aux paramètres** $(A, B, c, D, d)$, ordonnés avec $c \le d$ ; $A$, puis $D$, sont ramenés à **exactement 0** quand le modèle contraint à 0 (les autres paramètres ré-ajustés) est aussi bon que le modèle libre à `nr_tol` près.

### 2.3 Les coordonnées de l'optimiseur

L'optimiseur travaille sur des coordonnées **sans contrainte et d'ordre 1**, avec l'âge centré sur l'âge moyen des décès $\bar x$ ; on pose $b = B\,c^{\bar x}$ et $g = D\,d^{\bar x}$ :

$$A = s_A\,\theta_A^2,\quad b = e^{\theta_b},\quad c = e^{e^{\theta_c}},\quad g = \theta_g^2\,\frac{b\,w_{\max}}{v_{\max}},\quad d = e^{e^{\theta_d}}, \qquad w = c^{x-\bar x},\; v = d^{x-\bar x}.$$

- $A$ et $D$ (par $g$) sont des **carrés** : ils atteignent exactement 0, au lieu de dériver vers $10^{-17}$ pendant des milliers d'itérations.
- La double exponentielle garantit $c > 1$ et $d > 1$.
- Une unité de $\theta_g$ correspond à un second terme aussi grand que le premier à l'âge le plus élevé du départ.
- $B = b\,c^{-\bar x}$ et $D = g\,d^{-\bar x}$ sont rapportés.

Nelder-Mead s'arrête **près** d'un mur, jamais dessus (sur des données de type Makeham, il finit avec un $D$ minuscule et une base $d$ presque égale à $c$ : les deux termes presque confondus). C'est `refit_with_zero` qui tranche : si $A$ (ou $D$) à 0 coûte moins que `nr_tol`, le paramètre est rapporté égal à 0.0.

### 2.4 Plafond d'itérations

C'est `fit_common.NM_MAXITER` (2 000 par exécution) qui limite Nelder-Mead, **pas** `nr_maxiter`, qui plafonne des pas de Newton (200 par défaut). `nr_tol` est la tolérance sur l'objectif.

## 3. Détails et cas limites

- **Minimum** : 5 âges avec exposition et des décès à 2 âges distincts, sinon erreur en français.
- `converged` est vrai si Nelder-Mead a atteint ses tolérances et si une dernière relance n'a rien gagné ; `n_iter` somme les itérations de ses exécutions.
- **Avertissements dans `message`** : $D = 0$ (le second terme disparaît, Makeham 2ᵉ ordre se réduit à Makeham) ; $A = 0$ ; Nelder-Mead non convergé.
- **Paramètres rendus** : `A`, `B`, `c`, `D`, `d` (base des âges) et `n_params = 5`.

## 4. Forces et limites

**Forces.** C'est le modèle de zone II qui épouse le mieux la courbure de la mortalité adulte sur des données nationales : les hommes comme les femmes de la France 2020 le choisissent, avec un ΔAIC de plusieurs centaines à plusieurs milliers face aux autres lois.

**Limites.**

- **Cinq paramètres** : sur une petite population (zone II courte, peu de décès), il surajuste facilement ; le critère d'information le dit, mais seulement si on le consulte.
- Nelder-Mead est **plus lent** (plusieurs centaines d'itérations en deux exécutions) et ne garantit pas l'optimum global ; la graine de Makeham et la contrainte $c \le d$ rendent le résultat reproductible mais pas prouvé.
- Les paramètres sont **corrélés** et peu interprétables individuellement : on regarde la courbe, pas les coefficients.
- Comme toutes les lois de la zone II, elle est faite pour 20-85 ans : en dehors, ses exponentielles divergent.

## 5. Résultats sur la France 2020

Réglages par défaut (zone II : 20 à 85 ans).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | A = 0, B = 2.907e-05, c = 1.086, D = 1.62e-13, d = 1.354 | 66 | -468.1 | 946.2 | 0.0 | 0.000 | 53.0 % |
| Masculin | A = 0.0001049, B = 5.142e-05, c = 1.089, D = 2.11e-19, d = 1.586 | 66 | -556.6 | 1123.1 | 0.0 | 0.000 | 50.0 % |
<!-- resultats:fin -->

## Références

- Makeham, W.M. (1860). On the law of mortality. *Journal of the Institute of Actuaries*, 8.
- Nelder, J.A., Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4).
