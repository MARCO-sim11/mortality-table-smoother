# Polynôme de degré 3 (zone I)

> **En une phrase.** Le logarithme de la force de mortalité est ajusté par un polynôme du troisième degré, par moindres carrés pondérés : une courbe souple, à quatre paramètres, obtenue en une seule résolution linéaire.

| | |
|---|---|
| **Zone** | I : de l'âge 0 à $\omega_1$ (21 âges par défaut) |
| **Modèle** | $\ln\mu_x = a_0 + a_1 x + a_2 x^2 + a_3 x^3$ |
| **Estimation** | moindres carrés pondérés sur $\ln(D_x/E_x)$, poids $D_x$, forme fermée |
| **Paramètres** | 4 (`n_params = 4`) |
| **Itérations** | aucune (`n_iter = 0`) |
| **Code** | `src/zone1_poly.py`, fonction `fit_poly3` |

## 1. Pourquoi un polynôme en zone I

La mortalité des jeunes n'est pas monotone : pic à la naissance, chute très rapide dans la première année, minimum vers 8-12 ans, puis remontée (la « bosse » des jeunes adultes, liée aux accidents). Gompertz ou Makeham, qui ne savent que croître, ne conviennent pas ici. Un polynôme du **logarithme** du taux est une courbe souple, strictement positive (on repasse par l'exponentielle) et peu coûteuse : quatre paramètres.

## 2. Le modèle

$$\mu_x \;=\; \exp\!\big(a_0 + a_1 x + a_2 x^2 + a_3 x^3\big).$$

Un polynôme de degré 3 a au plus un minimum et un maximum locaux : il peut donc décrire la forme en « U » de l'enfance et de l'adolescence. Les coefficients n'ont pas d'interprétation directe : ce sont ceux d'une forme, pas de lois de mortalité. L'exponentielle garantit $\mu_x > 0$.

## 3. L'estimation pas à pas

**Étape 1 : la variable de travail.** Sur les âges utilisables (ceux avec $D_x > 0$ et $E_x > 0$), on pose $y_x = \ln(D_x/E_x)$, le logarithme du taux brut. Les âges sans décès n'ont pas de logarithme : ils sont écartés de l'ajustement.

**Étape 2 : les poids.** Si $D_x \sim \text{Poisson}(E_x\mu_x)$, alors par la méthode delta

$$\operatorname{Var}\big[\ln(D_x/E_x)\big] \;\approx\; \frac{1}{E_x\mu_x} \;\approx\; \frac{1}{D_x}.$$

Le bon poids d'un âge est l'inverse de cette variance, donc $w_x = D_x$ : l'enfance, où l'on observe des centaines de décès par âge, compte plus que les âges à quelques décès. Les moindres carrés pondérés corrigent ainsi l'hétéroscédasticité.

**Étape 3 : la résolution.** Avec la matrice $A$ de colonnes $[1, x, x^2, x^3]$ et $W = \operatorname{diag}(w_x)$ :

$$\hat a \;=\; (A^\top W A)^{-1} A^\top W\,y .$$

Le système $4\times 4$ est résolu par `numpy.linalg.solve`, jamais en inversant la matrice, jamais par `numpy.polyfit`.

**Étape 4 : le conditionnement.** Les puissances d'un âge ($x^3$ jusqu'à $10^6$) rendent $A^\top W A$ très mal conditionnée (jusqu'à $10^{18}$ sur 60-109 ans). On centre donc les âges et on les ramène à $[-1, 1]$ : $t = (x - c)/h$ avec $c$ le milieu et $h$ la demi-étendue des âges utilisables. On résout en $t$, puis on **repasse à la base des âges** (le polynôme en $x$) par le développement du binôme :

$$a_j \;=\; \sum_{k \ge j} b_k \binom{k}{j}\,\frac{(-c)^{k-j}}{h^k},$$

où les $b_k$ sont les coefficients obtenus en $t$. Mathématiquement rien ne change ; numériquement, on garde environ 14 chiffres partout.

**Étape 5 : la courbe.** $\hat\mu_x$ est le polynôme évalué en **chaque** âge fourni, y compris ceux écartés de l'ajustement.

**Étape 6 : la log-vraisemblance.** Les moindres carrés ne maximisent pas la vraisemblance de Poisson ; on la **calcule** ensuite sur la courbe obtenue (`fit_common.poisson_loglik`, avec les âges sans décès) pour que la zone I soit sur la même échelle que les modèles de Poisson des zones II et III. C'est ce qui rend l'AIC comparable.

### Lien avec la régression de Poisson

La régression de Poisson à lien logarithmique se résout par moindres carrés itérativement repondérés (IRLS) : poids égal au nombre de décès attendus $\hat E_x$, variable de travail $\ln\hat\mu_x + (D_x - \hat E_x)/\hat E_x$. En partant du modèle saturé ($\hat\mu_x = D_x/E_x$), la variable de travail vaut exactement $\ln(D_x/E_x)$ et le poids $D_x$. La méthode est donc **la première itération de l'IRLS** à partir du modèle saturé : on ne fait pas les suivantes, qui amélioreraient un peu la vraisemblance au prix d'une itération.

## 4. Détails et décisions d'implémentation

- **Âges sans décès ou sans exposition** : exclus de l'ajustement, mais la courbe leur donne quand même un taux.
- **Minimum** : 4 âges distincts avec $D > 0$ et $E > 0$ (4 coefficients), sinon une erreur en français est levée (« zone I trop courte, ou trop d'âges sans décès »).
- **Réglages** : aucun ($\omega_1$ détermine seulement les lignes reçues). `n_iter = 0`, et `converged` est vrai si la solution est finie.
- **Biais des petits effectifs** : $\mathbb{E}[\ln(D/E)]$ est biaisée vers le bas quand $D$ est petit (inégalité de Jensen). Les poids $D_x$ limitent l'effet de ces âges mais ne le suppriment pas.

## 5. Ce que renvoie la méthode

`params` contient `a0, a1, a2, a3` (base des âges) et `n_params = 4` ; `mu_hat` la courbe ; `loglik` la log-vraisemblance de Poisson ; `message` indique le nombre d'âges utilisés et exclus.

## 6. Forces et limites

**Forces.** Déterministe, instantané, sans itération ni échec numérique possible ; peu de paramètres ; la forme en U de l'enfance est reproduite.

**Limites.**

- Un polynôme de degré 3 ne suit pas la **chute entre 0 et 1 an** : sur la France 2020, le taux de l'âge 1 est surestimé d'un facteur 3,8 (femmes) à 5,4 (hommes) et ceux des âges 2 et 3 d'un facteur 1,6 à 2,1 ; la forme générale (chute, minimum vers 6-8 ans, remontée) reste correcte.
- Il ne peut pas reproduire à la fois le pic de la naissance et la bosse des jeunes adultes : ce sont plus d'extrema que degré 3 n'en permet.
- Hors de sa zone, il diverge très vite : le raccord ne l'évalue que dans la fenêtre de jonction autour de $\omega_1$.
- Avec des expositions nationales, les résidus restent grands (près de la moitié des âges à $|r| > 2$ ci-dessous) : le critère décisif est l'AIC, pas le $\chi^2$.

## 7. Résultats sur la France 2020

Réglages par défaut ($\omega_1 = 20$, 21 âges, de 0 à 20 ans). Le tableau est produit par le code (`docs/build_method_results.py`).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | a₀ = -5.833, a₁ = -1.415, a₂ = 0.1322, a₃ = -0.003449 | 21 | -218.0 | 444.1 | 0.0 | 0.063 | 52.4 % |
| Masculin | a₀ = -5.662, a₁ = -1.337, a₂ = 0.1211, a₃ = -0.002962 | 21 | -310.0 | 628.1 | 0.0 | 0.007 | 47.6 % |
<!-- resultats:fin -->

L'AIC se compare seulement entre méthodes d'une même zone (ΔAIC = écart au meilleur de la zone, ici le polynôme face à la spline naturelle).

## Références

- Méthode delta et poids de la régression sur $\ln(D/E)$ : McCullagh, P., Nelder, J.A. (1989). *Generalized Linear Models*, 2ᵉ éd. Chapman & Hall.
- Heligman, L., Pollard, J.H. (1980). The age pattern of mortality. *Journal of the Institute of Actuaries*, 107. (La « bosse » des jeunes adultes.)
