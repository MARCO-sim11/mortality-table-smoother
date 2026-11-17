# Évaluation et sélection des méthodes

> **En une phrase.** Chaque méthode est jugée sur la **même** base, celle des taux lissés qu'elle produit, avec la log-vraisemblance, l'AIC, le BIC, le $\chi^2$, la déviance, un test des séquences, les résidus et un indicateur de rugosité de Whittaker-Henderson ; une règle propose, par zone, la méthode à garder.

| | |
|---|---|
| **Entrée** | les `FitResult` d'un sexe et la table brute de ce sexe |
| **Sortie** | un tableau (une ligne par méthode **et par zone**), des séries de résidus, une suggestion par zone |
| **Réglage** | $h$ de Whittaker-Henderson (1 par défaut) |
| **Code** | `src/evaluator.py` : `evaluate_fits`, `suggest_selection`, `residual_series` |

## 1. Ce qui est comparé avec quoi

Pour chaque méthode, la courbe est relue **sur les âges de données de sa zone** (ceux avec $E > 0$) et la log-vraisemblance est **recalculée** avec la formule commune de Poisson, y compris pour la zone III dont la courbe va jusqu'à $\omega_3$. Toutes les méthodes sont donc sur une même échelle.

- **$p$** est le nombre de paramètres propre à la méthode (`n_params`) ; il dépend des nœuds pour les splines.
- **$n$** est le nombre de lignes de données avec $E > 0$ dans la zone.
- **L'AIC et le BIC ne se comparent qu'entre méthodes d'une même zone** (mêmes lignes) ; la colonne **ΔAIC** donne l'écart au meilleur AIC de la zone. La spline globale a une ligne dans chacune des trois zones.

## 2. Les critères

Avec $\hat E_x = E_x\hat\mu_x$ :

| Critère | Formule | Lecture |
|---|---|---|
| Log-vraisemblance | $\ell = \sum[D\ln(\hat E) - \hat E - \ln\Gamma(D+1)]$ | plus haut = mieux |
| AIC | $-2\ell + 2p$ | plus bas = mieux ; pénalise chaque paramètre de 2 |
| BIC | $-2\ell + p\ln n$ | plus bas = mieux ; pénalise plus fort quand $n$ est grand |
| $\chi^2$ de Pearson | $\sum (D-\hat E)^2/\hat E$ | à comparer à un $\chi^2$ à $n - p$ degrés de liberté |
| Déviance | $\sum d_x,\ d_x = 2[D\ln(D/\hat E) - (D - \hat E)]$ | écart au modèle saturé ; plus bas = mieux |
| Résidu de déviance | $r_x = \operatorname{signe}(D - \hat E)\sqrt{d_x}$ | environ $\mathcal N(0,1)$ si le modèle est juste : l'essentiel dans $[-2, +2]$ |
| Part $\lvert r\rvert > 2$ | part des âges hors de la bande | 5 % attendus si le modèle est juste |
| Test des séquences | voir ci-dessous | signes des résidus en alternance « aléatoire » ? |
| Whittaker-Henderson | voir ci-dessous | fidélité et rugosité |

Le $\chi^2$ n'est **pas fiable** là où $\hat E < 5$ (les âges les plus élevés de la zone III), comme tout $\chi^2$ sur des comptages épars. Sur des expositions nationales, les $p$-valeurs du $\chi^2$ sont proches de 0 pour **toutes** les lois paramétriques : les modèles à 2 ou 5 paramètres ne peuvent pas rendre compte de tous les détails de millions d'années-personnes ; on lit alors l'AIC, les résidus et le test des séquences plutôt que le seuil du $\chi^2$.

### Le test des séquences (Wald-Wolfowitz)

On regarde le **signe** des résidus dans l'ordre des âges : une séquence est un paquet de résidus de même signe. Avec $n_+$ résidus positifs, $n_-$ négatifs et $n = n_+ + n_-$, le nombre de séquences $R$ a, sous l'indépendance,

$$\mathbb E[R] = \frac{2n_+n_-}{n} + 1, \qquad \operatorname{Var}[R] = \frac{2n_+n_-\,(2n_+n_- - n)}{n^2\,(n-1)},$$

et $z = (R - \mathbb E[R])/\sqrt{\operatorname{Var}[R]}$ donne un $p$ bilatéral par l'approximation normale. **Trop peu de séquences** : des paquets d'âges entiers sont sur- ou sous-estimés, la forme de la loi est mauvaise. **Trop de séquences** : les résidus alternent plus que le hasard ne le ferait. Les résidus nuls sont écartés ; si tous ont le même signe, le test n'est pas défini ($p$ = `NaN`) et la méthode **ne passe pas** la règle « $p > 0{,}05$ ».

### Whittaker-Henderson

Le critère de Whittaker-Henderson d'une courbe lissée $z$ pour des taux bruts $q$ est

$$\mathrm{WH}(z) \;=\; \underbrace{\sum_x w_x\,(q_x - z_x)^2}_{\text{fidélité}} \;+\; h\,\underbrace{\sum_x \big(\Delta^2 z_x\big)^2}_{\text{rugosité}} .$$

Ici c'est un **diagnostic** d'une courbe déjà ajustée : $q$ et $z$ sont les $q$ bruts et lissés par âges consécutifs. Choix faits (la formule ne précise ni les poids ni $h$) : poids $w_x = E_x/\overline{E}$ (l'exposition, ramenée à sa moyenne pour que la fidélité ne dépende pas de la taille de la population), ordre 2 ($\Delta^2$, la différence seconde), $h = 1$ par défaut (réglable dans la page). Le tableau donne la fidélité et la rugosité **séparément** et leur somme à $h$ : seul leur rapport compte, et d'une méthode à l'autre la rugosité seule est le nombre comparable.

## 3. La règle de suggestion

Pour chaque zone, l'évaluateur propose :

1. d'écarter les méthodes dont l'AIC n'est pas fini, et celles qui n'ont pas convergé **tant qu'une autre de la zone a convergé** ;
2. parmi celles qui restent, **l'AIC le plus bas parmi les méthodes dont le test des séquences n'est pas significatif** ($p > 0{,}05$) ;
3. si aucune ne passe ce test, **l'AIC le plus bas** quand même (la colonne « Runs OK » dit que le test n'était pas passé).

Les menus de la page Évaluation sont préremplis avec cette suggestion ; l'utilisateur reste libre de choisir autrement.

## 4. Lire le tableau : la France 2020

Voici le tableau des huit méthodes de zone sur la France 2020 (réglages par défaut, sans la spline globale). Quelques lectures :

- **Zone I** : le polynôme l'emporte sur la spline naturelle (moins de paramètres, ΔAIC de 13 à 53 pour la spline) ; les deux ont des résidus grands (d'un tiers à deux tiers des âges hors de $\pm 2$).
- **Zone II** : Makeham 2ᵉ ordre est loin devant, de 600 à plus de 2 000 points d'AIC ; Gompertz, Makeham et Perks sont tous très loin. Perks vaut Makeham plus 2 (voir la fiche de Perks).
- **Zone III** : Beard a le meilleur AIC mais la règle écarte celui des femmes (séquences significatives, $p \approx 0{,}025$) au profit de Kannisto.

<!-- resultats:debut -->
| Sexe | Zone | Méthode | p | AIC | ΔAIC | p(runs) | \|r\| > 2 | Suggérée |
|---|---|---|---|---|---|---|---|---|
| Féminin | Zone I | Polynôme deg 3 | 4 | 444.1 | 0.0 | 0.063 | 52.4 % | oui |
| Féminin | Zone I | Splines naturelles | 5 | 456.9 | 12.8 | 0.133 | 33.3 % |  |
| Féminin | Zone II | Gompertz | 2 | 3163.7 | 2217.4 | 0.000 | 74.2 % |  |
| Féminin | Zone II | Makeham | 3 | 2739.2 | 1793.0 | 0.000 | 69.7 % |  |
| Féminin | Zone II | Makeham 2e ordre | 5 | 946.2 | 0.0 | 0.000 | 53.0 % | oui |
| Féminin | Zone II | Perks | 4 | 2741.2 | 1795.0 | 0.000 | 69.7 % |  |
| Féminin | Zone III | Kannisto | 2 | 617.9 | 6.0 | 0.545 | 24.0 % | oui |
| Féminin | Zone III | Beard | 3 | 611.8 | 0.0 | 0.025 | 32.0 % |  |
| Masculin | Zone I | Polynôme deg 3 | 4 | 628.1 | 0.0 | 0.007 | 47.6 % | oui |
| Masculin | Zone I | Splines naturelles | 5 | 680.7 | 52.6 | 0.014 | 61.9 % |  |
| Masculin | Zone II | Gompertz | 2 | 2124.9 | 1001.8 | 0.000 | 56.1 % |  |
| Masculin | Zone II | Makeham | 3 | 1746.0 | 622.9 | 0.000 | 56.1 % |  |
| Masculin | Zone II | Makeham 2e ordre | 5 | 1123.1 | 0.0 | 0.000 | 50.0 % | oui |
| Masculin | Zone II | Perks | 4 | 1748.0 | 624.9 | 0.000 | 56.1 % |  |
| Masculin | Zone III | Kannisto | 2 | 271.9 | 11.3 | 0.499 | 12.0 % |  |
| Masculin | Zone III | Beard | 3 | 260.5 | 0.0 | 0.336 | 12.0 % | oui |
<!-- resultats:fin -->

## Références

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6).
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2).
- Wald, A., Wolfowitz, J. (1940). On a test whether two samples are from the same population. *The Annals of Mathematical Statistics*, 11(2).
- Whittaker, E.T. (1922). On a new method of graduation. *Proceedings of the Edinburgh Mathematical Society*, 41.
- Henderson, R. (1924). A new method of graduation. *Transactions of the Actuarial Society of America*, 25.
