# Kannisto (zone III)

> **En une phrase.** Aux âges les plus élevés, la mortalité cesse de doubler tous les huit ans et se **tasse** : Kannisto la décrit par une logistique, $\mu_x = \dfrac{a\,e^{bt}}{1 + a\,e^{bt}}$ avec $t = x - \omega_2$, de plateau 1, ajustée par une grille de départ puis un Newton-Raphson $2\times 2$ sur la vraisemblance de Poisson.

| | |
|---|---|
| **Zone** | III : de $\omega_2$ à $\omega_3$ (85 à 110 ans par défaut) |
| **Modèle** | $\mu_x = \operatorname{sigmoïde}(\ln a + b\,t) = \dfrac{a\,e^{bt}}{1 + a\,e^{bt}}$, $t = x - \omega_2$, $a, b > 0$ |
| **Estimation** | maximum de vraisemblance de Poisson : grille sur $(\ln a, b)$, puis Newton-Raphson $2\times 2$ |
| **Paramètres** | 2 (`n_params = 2`) |
| **Plateau** | 1 ; Gompertz tant que $\mu_x \ll 1$ |
| **Code** | `src/zone3.py`, fonction `fit_kannisto` |

## 1. Pourquoi une loi propre au grand âge

Au-delà de 85 ans, il reste peu d'exposés, les décès se comptent en dizaines, et la mortalité **décélère** : elle croît moins vite que l'exponentielle de Gompertz. Kannisto (1994) propose une forme qui se comporte comme Gompertz tant que le taux est petit, puis sature.

$$\mu_x \;=\; \frac{a\,e^{bt}}{1 + a\,e^{bt}}, \qquad t = x - \omega_2 \qquad\Longleftrightarrow\qquad \frac{\mu_x}{1 - \mu_x} = a\,e^{bt}.$$

Le **logit** de $\mu_x$ est une droite de l'âge : $\operatorname{logit}\mu_x = \ln a + b\,t$.

- $a$ est la **cote** $\mu/(1-\mu)$ à l'âge $\omega_2$ (de l'ordre de 0,06 à 0,10 sur la France 2020).
- $b$ est la vitesse de croissance de cette cote (autour de 0,16).
- Le taux est borné par 1 : plateau à 1 par an.

**L'origine de l'âge.** Une recherche naturelle porte sur $a \in [0{,}001\,;\,1]$ et $b \in [0{,}05\,;\,0{,}20]$. Avec les âges absolus, le $a$ optimal vaudrait $7\cdot 10^{-8}$ : aucun point de la grille n'en approche. En comptant l'âge depuis $\omega_2$ ($t = x - \omega_2$), $a$ devient la cote à $\omega_2$ et la grille encadre l'optimum. Le $a$ « en âge absolu » de la littérature est $a\,e^{-b\,\omega_2}$ ; `params['x0']` donne l'origine $\omega_2$.

## 2. Vraisemblance de Poisson, pas de Bernoulli

Une vraisemblance de Bernoulli ($D \sim \text{Binomiale}(E, \mu)$ approchée) serait le choix naturel. Mais $D_x > E_x$ se produit aux âges extrêmes (France 2020 : 105 ans pour les deux sexes, 107 à 109 ans pour les hommes, taux jusqu'à 2,0) : une binomiale y a un nombre négatif de survivants et aucune vraisemblance. On prend donc $D_x \sim \text{Poisson}(E_x\mu_x)$, la vraisemblance de toutes les autres zones (l'AIC reste sur une seule échelle) ; ce sont d'ailleurs des gradients de Poisson, $(D - E\mu)(1-\mu)$, que l'estimation ci-dessous utilise.

## 3. L'estimation pas à pas

### 3.1 La grille de départ

La log-vraisemblance de cette sigmoïde n'est **pas concave** et présente des régions plates : un Newton parti au hasard peut se perdre dans la zone saturée. On cherche donc d'abord le meilleur point d'une grille de `grid_n` $\times$ `grid_n` points (100 × 100 par défaut) sur $\ln a \in [-10, 0]$ et $b \in [0{,}05\,;\,0{,}20]$, en minimisant la demi-déviance.

### 3.2 Newton-Raphson $2\times 2$ en $(\ln a, b)$

Avec $\eta = \ln a + b\,t$ et $\mu = \operatorname{sigmoïde}(\eta)$, le score est

$$\frac{\partial\ell}{\partial\ln a} = \sum_x (D_x - E_x\mu_x)(1 - \mu_x), \qquad \frac{\partial\ell}{\partial b} = \sum_x t_x\,(D_x - E_x\mu_x)(1-\mu_x),$$

et l'information observée est $-H = \sum_x w_x \begin{pmatrix}1 & t_x\\ t_x & t_x^2\end{pmatrix}$ avec $w_x = \mu_x(1-\mu_x)(D_x + E_x - 2E_x\mu_x)$.

Contrairement à Gompertz, $w_x$ **devient négatif** près du plateau ($\mu > (1 + D/E)/2$) : la hessienne observée n'est plus définie. Là où elle ne l'est pas, on utilise l'**information attendue** (scoring de Fisher) $w_x = E_x\mu_x(1-\mu_x)^2 > 0$, toujours valide. Le nombre de ces pas est rapporté dans `message`.

Trois protections complètent l'itération :

- **Arrêt** comme pour Gompertz : écart de log-vraisemblance restant $\tfrac12\,g^\top(-H)^{-1}g \le$ `nr_tol`, au plus `nr_maxiter` pas.
- **Pas réduit de moitié** tant que la demi-déviance monte.
- **Région de confiance** : près du plateau, le score et l'information tendent tous deux vers 0 avec $(1 - \mu)$ et leur rapport explose ; un pas complet peut projeter l'itéré dans la région saturée ($\mu = 1$ à tous les âges) d'où il ne revient pas, la réduction de moitié demandant seulement que la déviance ne monte pas. Le pas est donc réduit pour qu'aucun logit ne bouge de plus de 4. Depuis le meilleur point de la grille, ce cas ne se produit jamais ; il protège les départs quelconques.

L'affinage est **sans contrainte** : un $b$ hors de $[0{,}05\,;\,0{,}20]$ est trouvé si les données le veulent.

## 4. Ce que couvre le résultat

Le résultat de la zone III couvre **toujours** $\omega_2, \ldots, \omega_3$ : `ages` va de $\omega_2$ à $\omega_3$ et `mu_hat` donne la courbe sur tous ces âges, **extrapolée** au-delà de la dernière ligne de données (une table HMD s'arrête à 109 ans). Les lignes reçues servent à l'ajustement et au calcul de `loglik`.

## 5. Détails et cas limites

- Tous les âges avec $E_x > 0$ entrent dans la vraisemblance, âges sans décès compris.
- **Minimum** : 2 âges avec exposition et des décès à 2 âges distincts ; sinon, ou si aucun point de la grille n'est admissible, erreur en français.
- **Avertissement** : $b \le 0$ (la mortalité ne croît pas avec l'âge sur la zone).
- **Paramètres rendus** : `a` (cote à $\omega_2$), `b`, `x0` ($= \omega_2$) et `n_params = 2` ; `n_iter` compte les pas de Newton (2 sur la France 2020 depuis la grille).

## 6. Forces et limites

**Forces.** Deux paramètres seulement : peu de risque de surajustement avec si peu de décès ; borné par 1, donc sans extrapolation explosive ; la grille de départ rend l'ajustement fiable.

**Limites.**

- Le plateau est **imposé à 1** : si la vraie mortalité sature plus haut ou plus bas, Kannisto ne peut pas s'y adapter (c'est l'objet de Beard, qui libère le plateau). Sur les hommes de la France 2020, les taux observés dépassent 1 et atteignent 2,0.
- Les derniers âges ont des exposés de quelques unités : la forme de la courbe entre 105 et 110 ans repose sur très peu d'information.
- L'extrapolation au-delà de la dernière ligne de données est une hypothèse de forme, pas une mesure.

## 7. Résultats sur la France 2020

Réglages par défaut (zone III : 85 à 110 ans ; les données s'arrêtent à 109, soit 25 âges ajustés).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | a = 0.06152, b = 0.1606 | 25 | -306.9 | 617.9 | 6.0 | 0.545 | 24.0 % |
| Masculin | a = 0.1004, b = 0.1556 | 25 | -133.9 | 271.9 | 11.3 | 0.499 | 12.0 % |
<!-- resultats:fin -->

## Références

- Kannisto, V. (1994). *Development of oldest-old mortality, 1950-1990: evidence from 28 developed countries*. Odense University Press.
- Thatcher, A.R., Kannisto, V., Vaupel, J.W. (1998). *The Force of Mortality at Ages 80 to 120*. Odense University Press.
