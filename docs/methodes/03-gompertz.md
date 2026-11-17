# Gompertz (zone II)

> **En une phrase.** La force de mortalité croît exponentiellement avec l'âge, $\mu_x = B\,c^x$ : deux paramètres, estimés par maximum de vraisemblance de Poisson avec un Newton-Raphson $2\times 2$ dont la vraisemblance est strictement concave.

| | |
|---|---|
| **Zone** | II : de $\omega_1$ à $\omega_2$ (20 à 85 ans par défaut) |
| **Modèle** | $\mu_x = B\,c^x = \exp(\alpha + \beta x)$, avec $\alpha = \ln B$, $\beta = \ln c$ |
| **Estimation** | maximum de vraisemblance de Poisson, Newton-Raphson $2\times 2$ |
| **Paramètres** | 2 (`n_params = 2`) |
| **Départ** | régression MCO de $\ln(D/E)$ sur l'âge |
| **Code** | `src/zone2_gompertz.py`, fonction `fit_gompertz` |

## 1. La loi

Gompertz (1825) observe que, à l'âge adulte, le taux de mortalité est multiplié par un facteur à peu près constant d'une année à l'autre. En logarithme, la mortalité est donc une **droite** de l'âge :

$$\ln\mu_x = \alpha + \beta x \quad\Longleftrightarrow\quad \mu_x = B\,c^x .$$

- $\beta = \ln c$ est la **vitesse de vieillissement** : la mortalité augmente de $100\,(c-1)\,\%$ par année d'âge. Sur la France 2020, $c \approx 1{,}10$ (femmes) et $1{,}09$ (hommes).
- Le **temps de doublement** de la mortalité est $\ln 2/\beta$ : 7,3 ans pour les femmes, 8,0 ans pour les hommes sur la France 2020.
- $B = e^\alpha$ est le taux « extrapolé à l'âge 0 » : il n'a pas de sens démographique (la loi ne vaut pas à la naissance), c'est le niveau de la droite.

## 2. L'estimation

### 2.1 Pourquoi le maximum de vraisemblance de Poisson

Régresser $\ln(D/E)$ sur $x$ par moindres carrés ordinaires donnerait la même loi, mais : la variance de $\ln(D/E)$ vaut $\approx 1/D$ (les âges peu observés pèseraient autant que les autres), un âge sans décès n'a pas de logarithme, et les comptages sont réellement de Poisson. La régression MCO n'est conservée **que pour le point de départ** de l'itération.

La log-vraisemblance (sans les termes qui ne dépendent pas de $\theta = (\alpha, \beta)$) est

$$\ell(\alpha,\beta) \;=\; \sum_x \Big[\, D_x(\alpha + \beta x) \;-\; E_x\,e^{\alpha+\beta x}\Big].$$

### 2.2 Gradient et hessienne

Avec $\mu_x = e^{\alpha+\beta x}$ :

$$\frac{\partial\ell}{\partial\alpha} = \sum_x (D_x - E_x\mu_x), \qquad \frac{\partial\ell}{\partial\beta} = \sum_x x\,(D_x - E_x\mu_x),$$

$$H \;=\; -\sum_x E_x\mu_x \begin{pmatrix} 1 & x \\ x & x^2 \end{pmatrix}.$$

La hessienne est **définie négative** dès que deux âges distincts ont $E_x\mu_x > 0$ : $\ell$ est strictement **concave**. Il existe donc un seul maximum, il existe dès que des décès sont observés à deux âges au moins, et un pas de Newton pointe toujours vers le haut.

### 2.3 L'algorithme

1. **Départ** : $(\alpha_0, \beta_0)$ = régression MCO de $\ln(D_x/E_x)$ sur $x$, sur les âges avec $D_x > 0$ (non pondérée).
2. **Itérer** $\theta \leftarrow \theta + (-H)^{-1} g$ avec le gradient $g$ et la hessienne ci-dessus.
3. **Arrêt** quand l'écart de log-vraisemblance restant, estimé par $\tfrac12\,g^\top(-H)^{-1}g$ (décrémentation de Newton), passe sous `nr_tol`, ou après `nr_maxiter` pas.
4. **Résultat** : $B = e^{\alpha}$, $c = e^{\beta}$.

Deux décisions d'implémentation :

- **Règle d'arrêt.** La décrémentation de Newton est l'écart de log-vraisemblance qui reste à gagner, au second ordre. Elle ne dépend pas de l'échelle des âges, à la différence de la norme du gradient (dont la composante en $\beta$ est multipliée par $x$).
- **Pas réduit de moitié.** Un pas complet qui ferait baisser $\ell$ (ou déborder l'exponentielle) est divisé par deux jusqu'à ne plus la faire baisser. Depuis le départ MCO, cela ne se déclenche presque jamais : c'est un garde-fou.

## 3. Détails et cas limites

- Tous les âges avec $E_x > 0$ entrent dans la vraisemblance, **âges sans décès compris** (terme $-E_x\mu_x$). Les âges sans exposition ou à valeur manquante sont ignorés, mais la courbe leur donne quand même un taux.
- **Minimum** : 2 âges distincts avec décès (sinon erreur en français : « zone II trop courte, ou presque aucun décès »).
- **Convergence** : en pratique 3 pas de Newton sur la France 2020 depuis le départ MCO. Si l'itération s'arrête sans converger (plafond atteint, hessienne singulière, aucun pas ne fait monter $\ell$), le dernier point est rendu avec `converged = False` et la raison dans `message`.
- **Avertissement** : si $\beta \le 0$, la mortalité ne croît pas avec l'âge sur la zone et la loi de Gompertz est peu pertinente.
- **Paramètres rendus** : `alpha`, `beta`, `B`, `c` et `n_params = 2`.

## 4. Forces et limites

**Forces.** Un modèle interprétable en deux nombres (niveau, vitesse de vieillissement) ; vraisemblance concave, donc un seul maximum et une convergence en quelques pas ; c'est la brique de base de toutes les autres lois de la zone II.

**Limites.**

- Elle ignore la mortalité « accidentelle » indépendante de l'âge : à 20-40 ans la droite surestime ou sous-estime selon les données, ce qui est corrigé par Makeham.
- Elle impose une pente **constante** au logarithme de la mortalité : si cette pente varie avec l'âge, la droite ne peut pas la suivre, et les résidus deviennent systématiques (c'est ce que montre le test des séquences sur la France 2020).
- Avec les expositions de la France entière, c'est le plus mauvais des modèles de la zone à l'AIC (voir le tableau).

## 5. Résultats sur la France 2020

Réglages par défaut (zone II : 20 à 85 ans, 66 âges).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | α = -11.17, β = 0.09502, B = 1.414e-05, c = 1.1 | 66 | -1579.8 | 3163.7 | 2217.4 | 0.000 | 74.2 % |
| Masculin | α = -9.956, β = 0.08708, B = 4.744e-05, c = 1.091 | 66 | -1060.5 | 2124.9 | 1001.8 | 0.000 | 56.1 % |
<!-- resultats:fin -->

## Références

- Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. *Philosophical Transactions of the Royal Society*, 115.
- Richards, S.J. (2012). A handbook of parametric survival models for actuarial use. *Scandinavian Actuarial Journal*.
