# Cadre général du lissage

> **En une phrase.** Chaque âge apporte un nombre de décès $D_x$ pour une exposition $E_x$ ; on cherche, zone d'âge par zone d'âge, une courbe $\mu_x$ qui rend ces données vraisemblables, puis on raccorde les trois zones en une seule table de mortalité.

Ce document pose ce que **toutes** les méthodes ont en commun : les données, le modèle statistique, les deux familles d'estimateurs, les conventions de sortie. Les fiches suivantes détaillent chaque méthode, puis le raccord des zones, l'évaluation et la table de vie.

## 1. Les données et les notations

| Symbole | Sens | Remarque |
|---|---|---|
| $x$ | âge (en années entières) | une ligne par âge et par sexe |
| $D_x$ | nombre de décès à l'âge $x$ | entier ou non : les décès reconstruits à partir de taux ne sont pas entiers |
| $E_x$ | exposition centrale au risque | « nombre d'années-personnes » vécues à l'âge $x$ |
| $m_x = D_x / E_x$ | taux brut (`mx_raw`) | estimation directe, mais bruitée |
| $\mu_x$ | force de mortalité | taux instantané ; la courbe que l'on cherche |
| $q_x = 1 - e^{-\mu_x}$ | probabilité de décès dans l'année | déduite de $\mu_x$ à la fin (table de vie) |
| $\omega_1,\ \omega_2,\ \omega_3$ | frontières de zones et âge limite | 20, 85 et 110 par défaut ; $\omega_2 \ge \omega_1 + 1$ |

Le taux brut est l'estimateur « naturel » de $\mu_x$, mais son erreur relative vaut environ $1/\sqrt{D_x}$ : négligeable à 60 ans (des dizaines de milliers de décès), énorme à 108 ans (quelques décès). **Lisser, c'est remplacer ce nuage de points par une courbe qui ne garde que la forme générale.**

## 2. Le modèle statistique commun : Poisson

Toutes les méthodes reposent sur la même hypothèse : les décès des différents âges sont indépendants et

$$D_x \sim \text{Poisson}(E_x\,\mu_x), \qquad \mathbb{E}[D_x] = \operatorname{Var}[D_x] = E_x\,\mu_x .$$

La **log-vraisemblance** d'une courbe $\mu$ est donc

$$\ell(\mu) \;=\; \sum_x \Big[\, D_x \ln(E_x \mu_x) \;-\; E_x \mu_x \;-\; \ln \Gamma(D_x + 1) \Big].$$

Le dernier terme ne dépend pas de $\mu$ ; il est conservé pour que `loglik` soit une vraie log-vraisemblance, **comparable d'une méthode à l'autre** (c'est ce que l'AIC et le BIC exigent). Il utilise $\ln\Gamma(D+1)$ et non $\ln D!$, ce qui accepte des décès non entiers. Un âge avec $D_x = 0$ contribue $-E_x\mu_x$ : il informe (« à ce taux, on aurait dû voir des décès »). Un âge sans exposition ($E_x \le 0$) ou avec une valeur manquante ne contribue à rien.

**Pourquoi Poisson, et pas les moindres carrés ni la binomiale ?**

- Les moindres carrés sur $\ln(D/E)$ supposent une variance constante ; or $\operatorname{Var}[\ln(D/E)] \approx 1/D$ : les âges peu observés pèseraient autant que les autres. Et $\ln 0$ n'existe pas.
- Le cahier des charges écrit une vraisemblance binomiale (Bernoulli) en zone III. Mais aux âges extrêmes $D_x$ **dépasse** $E_x$ (France 2020 : âge 105 pour les deux sexes, taux jusqu'à 2,0) : une binomiale n'y a plus de vraisemblance. Poisson n'a pas cette limite et donne une seule échelle pour toutes les zones.

### Déviance et demi-déviance

La **déviance** mesure l'écart à un modèle qui collerait parfaitement aux données (le modèle saturé, $\mu_x = D_x/E_x$) :

$$d_x \;=\; 2\Big[D_x \ln\frac{D_x}{\hat E_x} \;-\; (D_x - \hat E_x)\Big], \qquad \hat E_x = E_x\hat\mu_x ,$$

le premier terme valant 0 quand $D_x = 0$. La **demi-déviance** $\tfrac12\sum_x d_x$ est la log-vraisemblance à signe inversé, ancrée au modèle saturé : elle vaut 0 si la courbe colle aux taux bruts, et la courbe qui la minimise est exactement l'estimateur du maximum de vraisemblance. Les optimiseurs minimisent **elle** plutôt que $-\ell$, dont l'ordre de grandeur ($10^6$ pour une table nationale) noierait la tolérance dans le bruit d'arrondi.

## 3. Trois zones, des modèles différents

La mortalité n'obéit pas à une seule loi sur toute la vie. Les âges sont donc découpés en trois zones, les bornes étant **incluses des deux côtés** (un âge frontière appartient aux deux zones voisines) :

| Zone | Âges | Ce qui s'y passe | Méthodes |
|---|---|---|---|
| I | $0 \ldots \omega_1$ | pic de la naissance, chute, minimum vers 8-12 ans, « bosse » des jeunes adultes : non monotone | polynôme de degré 3, spline naturelle |
| II | $\omega_1 \ldots \omega_2$ | croissance exponentielle de la mortalité avec l'âge | Gompertz, Makeham, Makeham 2ᵉ ordre, Perks |
| III | $\omega_2 \ldots \omega_3$ | peu d'exposés, décès en dizaines, la hausse **ralentit** | Kannisto, Beard, Coale-Kisker |
| toutes | $0 \ldots \omega_3$ | une seule courbe sur tous les âges (option) | spline globale |

**Cas de la zone III.** Une table HMD s'arrête à 109 ans alors que $\omega_3 = 110$ par défaut ; c'est précisément l'extrapolation que cette zone fournit. Le résultat d'une méthode de zone III couvre donc **toujours** les âges $\omega_2, \ldots, \omega_3$, quelles que soient les lignes fournies : celles-ci servent seulement à ajuster la courbe et à calculer `loglik`.

## 4. Les méthodes d'un coup d'œil

| Méthode | Zone | Modèle | Estimation | $k$ |
|---|---|---|---|---|
| Polynôme degré 3 | I | $\ln\mu_x = a_0 + a_1x + a_2x^2 + a_3x^3$ | moindres carrés pondérés, forme fermée | 4 |
| Spline naturelle | I | spline cubique naturelle de $\ln\mu_x$, un nœud tous les 5 ans | moindres carrés pondérés, forme fermée | nœuds |
| Gompertz | II | $\mu_x = B\,c^x = e^{\alpha+\beta x}$ | Poisson, Newton-Raphson 2×2 | 2 |
| Makeham | II | $\mu_x = A + B\,c^x$ | Poisson, profil sur $c$ + Newton-Raphson | 3 |
| Makeham 2ᵉ ordre | II | $\mu_x = A + B\,c^x + D\,d^x$ | Poisson, Nelder-Mead | 5 |
| Perks | II | $\mu_x = (A + B\,c^x)/(1 + D\,c^x)$ | Poisson, Nelder-Mead | 4 |
| Kannisto | III | $\mu_x = \dfrac{a\,e^{bt}}{1 + a\,e^{bt}},\ t = x-\omega_2$ | Poisson, grille + Newton-Raphson | 2 |
| Beard | III | $\mu_x = \dfrac{B\,c^x}{1 + K\,B\,c^x}$ | Poisson, Nelder-Mead | 3 |
| Coale-Kisker | III | $q_x = q_{\omega_2}\,e^{k(x-\omega_2)}$, $q_{\omega_3} = 0{,}6$ | formule directe | 1 |
| Spline globale | toutes | spline naturelle de $\ln\mu_x$ sur $0\ldots\omega_3$ | moindres carrés pondérés | nœuds |

Le nombre de paramètres $k$ est celui que l'AIC et le BIC comptent (`params['n_params']`).

## 5. Deux familles d'estimateurs

### 5.1 Moindres carrés pondérés (zone I)

On régresse $y_x = \ln(D_x/E_x)$ sur une base de fonctions de l'âge avec les poids $w_x = D_x$ : solution de forme fermée, un seul système linéaire à résoudre, pas d'itération. C'est une approximation de l'estimateur de Poisson (voir la fiche du polynôme), choisie parce qu'elle ne peut pas échouer.

### 5.2 Maximum de vraisemblance de Poisson (zones II et III)

On maximise $\ell(\theta)$ sur les paramètres $\theta$ du modèle.

**Newton-Raphson** (Gompertz, le sous-problème $(A, B)$ de Makeham, Kannisto). À chaque pas, on corrige $\theta \leftarrow \theta + (-H)^{-1} g$, où $g$ est le gradient (« score ») et $H$ la hessienne de $\ell$, tous deux calculés analytiquement. Trois règles communes :

- **Arrêt** : on estime l'écart de log-vraisemblance qui reste à gagner, $\tfrac12\, g^\top(-H)^{-1} g$ (la « décrémentation de Newton »), et on s'arrête quand il passe sous `nr_tol` (1e-8 par défaut). Contrairement à la norme du gradient, ce critère ne dépend pas de l'échelle des paramètres.
- **Pas réduit de moitié** : si un pas complet fait baisser $\ell$ (ou fait déborder une exponentielle), on le divise par deux jusqu'à ce que $\ell$ ne baisse plus (jusqu'à 60 fois). C'est un garde-fou, pas un changement de méthode.
- **Plafond** : `nr_maxiter` pas au plus (200 par défaut) ; sinon `converged = False` et la raison est écrite dans `message`.

**Nelder-Mead** (Makeham 2ᵉ ordre, Perks, Beard). Méthode du simplexe, sans dérivée, qui convient aux formes fractionnaires dont la hessienne est mal conditionnée. Elle est pilotée par `fit_common.nelder_mead` :

- l'objectif est la **demi-déviance** ;
- le simplexe initial est construit autour d'un point de départ issu d'un modèle plus simple (Makeham pour Makeham 2 et Perks, Kannisto pour Beard), chaque coordonnée décalée d'environ 10 % ;
- une exécution peut s'arrêter sur un simplexe écrasé avant le minimum : on **relance** depuis le meilleur point (simplexe de côté 0,05) tant que cela améliore l'objectif de plus de la tolérance, au plus 5 fois ; chaque exécution est plafonnée à 2 000 itérations ;
- les paramètres sont recodés en coordonnées **sans contrainte et d'ordre 1** (logarithmes, doubles exponentielles pour les bases $c > 1$) ; les paramètres qui ont le droit de valoir 0 ($A$, $D$, $K$) sont codés par un **carré**, pour pouvoir atteindre 0 exactement ;
- un paramètre n'est jamais rendu « presque nul » : `refit_with_zero` ré-ajuste le modèle contraint à 0 (les autres paramètres libres) et, s'il est aussi bon que le modèle libre à `nr_tol` près, le paramètre est rapporté **exactement égal à 0**. Le rapport dit alors ce que les données soutiennent (par exemple « Perks se réduit à Makeham »).

## 6. Modèles emboîtés

Plusieurs modèles en contiennent d'autres comme cas particuliers :

$$\text{Gompertz} \;\subset\; \text{Makeham} \;\subset\; \{\text{Makeham 2}^e\text{ ordre},\ \text{Perks}\}, \qquad \text{Kannisto} = \text{Beard}\ (K=1), \qquad \text{Beard}\ (K=0) = \text{Gompertz}.$$

Conséquences : sur les mêmes âges, la log-vraisemblance maximale ne peut pas baisser quand on passe au modèle plus large ; et si le modèle large retombe sur le petit (paramètre à 0), l'AIC du large est celui du petit **plus 2 par paramètre inutile**. C'est ce qui arrive à Perks sur la France 2020 : son AIC vaut celui de Makeham plus 2.

## 7. Ce que renvoie chaque méthode

Toutes les fonctions `fit_<clé>(ages, Dx, Ex, cfg)` rendent un même objet `FitResult` :

| Champ | Contenu |
|---|---|
| `method`, `zone` | clé de la méthode et zone (`zone1`, `zone2`, `zone3`, ou `all` pour la spline globale) |
| `ages` | les âges couverts (ceux de la zone ; $\omega_2\ldots\omega_3$ en zone III) |
| `mu_hat` | la courbe lissée, en **chaque** âge de `ages`, y compris ceux ignorés par l'ajustement |
| `params` | les paramètres estimés (base « âge simple » du cahier des charges) et `n_params` |
| `loglik` | la log-vraisemblance de Poisson complète de la courbe sur les lignes fournies |
| `converged`, `n_iter` | l'arrêt a-t-il eu lieu normalement, après combien de pas ou d'itérations |
| `message` | un diagnostic en français (itérations, avertissements) |

Une méthode qui ne peut pas être ajustée (trop peu d'âges avec décès, par exemple) lève une erreur en français : l'application l'affiche comme une ligne « erreur » du tableau d'estimation sans arrêter les autres méthodes. Une méthode qui s'arrête sans converger rend quand même son dernier point, avec `converged = False`.

## 8. Les réglages qui agissent

| Réglage | Défaut | Agit sur |
|---|---|---|
| $\omega_1$, $\omega_2$, $\omega_3$ | 20, 85, 110 | les âges de chaque zone ; $\omega_2$ part toujours de $\omega_1 + 1$ |
| `nr_maxiter` | 200 | plafond des pas de Newton-Raphson (pas de Nelder-Mead) |
| `nr_tol` | 1e-8 | écart de log-vraisemblance toléré (Newton) ; tolérance de l'objectif (Nelder-Mead) |
| `grid_n` | 100 | taille de la grille de départ de Kannisto ($n \times n$ points) |
| $\sigma$ | 3 | demi-largeur de la fenêtre de raccord (voir la fiche du raccord) |
| $h$ | 1 | poids de la rugosité dans Whittaker-Henderson (voir la fiche de l'évaluation) |

## 9. Pour continuer

Une fiche par méthode (polynôme, spline, Gompertz, Makeham, Makeham 2ᵉ ordre, Perks, Kannisto, Beard, Coale-Kisker, spline globale), puis le raccord des zones, l'évaluation et la sélection, et la construction de la table de vie.

## Références

- McCullagh, P., Nelder, J.A. (1989). *Generalized Linear Models*, 2ᵉ éd. Chapman & Hall. (Régression de Poisson, déviance.)
- Nelder, J.A., Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4).
- Richards, S.J. (2012). A handbook of parametric survival models for actuarial use. *Scandinavian Actuarial Journal*.
