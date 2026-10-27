# Le raccord des trois zones

> **En une phrase.** Les trois courbes de zones sont fondues en une seule $\mu_x$ sur $0\ldots\omega_3$ par une pondération en tangente hyperbolique, de largeur $\pm 2\sigma$ autour de chaque frontière, dont les poids somment à 1 à tout âge.

| | |
|---|---|
| **Entrée** | une courbe par zone (une méthode choisie dans chaque zone) |
| **Sortie** | $\mu_x$ pour $x = 0, 1, \ldots, \omega_3$ : un tableau de longueur $\omega_3 + 1$ |
| **Réglage** | $\sigma$ (3 par défaut, de 1 à 10) ; fenêtre $[\omega - 2\sigma,\ \omega + 2\sigma]$ |
| **Code** | `src/joiner.py` : `blend_weight`, `join_zones`, `evaluate_curve`, `zone_values` |

## 1. Le problème

Juxtaposer trois courbes laisserait une marche à chaque frontière : le taux de la fin de la zone I n'est pas celui du début de la zone II. On les **fond** donc : près d'une frontière $\omega$, la table est une moyenne pondérée des deux courbes voisines, dont le poids passe progressivement de l'une à l'autre.

## 2. Le poids d'une frontière

Avec $u = (\omega - x)/\sigma$, le poids de la zone **à gauche** de $\omega$ est

$$w(x) \;=\; \begin{cases} 1 & u > 2,\\[2pt] \dfrac12\Big(1 + \dfrac{\tanh u}{\tanh 2}\Big) & |u| \le 2,\\[6pt] 0 & u < -2,\end{cases}$$

et la zone de droite pèse $1 - w$. Propriétés : $w = \tfrac12$ en $\omega$ ; $w(\omega + \delta) = 1 - w(\omega - \delta)$ ; la fenêtre fait $\pm 2\sigma$ ($\pm 6$ ans pour $\sigma = 3$).

### Bords continus (décision signalée)

La formule littérale du cahier des charges est $\tfrac12(1 + \tanh u)$. Mais en $|u| = 2$, elle vaut 0,982 et non 1 : si l'on coupe la fenêtre à cet endroit, on laisse une **marche de 1,8 %** de l'écart entre les deux zones à chaque bord. La version retenue **étale** la tangente dans la fenêtre pour qu'elle atteigne exactement 1 et 0 aux bords (le facteur $1/\tanh 2$) : même fenêtre, même milieu ($w = \tfrac12$), mais continue (avec une rupture de pente de 1,2 % de l'écart par âge pour $\sigma = 3$). La différence ne compte que pour une fenêtre large ($\sigma \ge 5$) : pour $\sigma = 3$, la plus grande irrégularité du poids est la raideur du milieu de la fenêtre, et les deux formules s'y valent. La constante `CONTINUOUS_EDGES` de `src/joiner.py` rétablit la formule littérale.

Valeurs pour $\sigma = 3$ (poids de la zone de gauche, $\delta = x - \omega$) :

<!-- resultats:debut -->
| x − ω | u = (ω − x) / σ | w, formule littérale | w retenu (bords continus) | 1 − w retenu |
|---|---|---|---|---|
| -7 | +2.333 | 0.9907 | 1.0000 | 0.0000 |
| -6 | +2.000 | 0.9820 | 1.0000 | 0.0000 |
| -5 | +1.667 | 0.9656 | 0.9829 | 0.0171 |
| -4 | +1.333 | 0.9350 | 0.9513 | 0.0487 |
| -3 | +1.000 | 0.8808 | 0.8950 | 0.1050 |
| -2 | +0.667 | 0.7914 | 0.8023 | 0.1977 |
| -1 | +0.333 | 0.6608 | 0.6668 | 0.3332 |
| 0 | +0.000 | 0.5000 | 0.5000 | 0.5000 |
| 1 | -0.333 | 0.3392 | 0.3332 | 0.6668 |
| 2 | -0.667 | 0.2086 | 0.1977 | 0.8023 |
| 3 | -1.000 | 0.1192 | 0.1050 | 0.8950 |
| 4 | -1.333 | 0.0650 | 0.0487 | 0.9513 |
| 5 | -1.667 | 0.0344 | 0.0171 | 0.9829 |
| 6 | -2.000 | 0.0180 | 0.0000 | 1.0000 |
| 7 | -2.333 | 0.0093 | 0.0000 | 1.0000 |
<!-- resultats:fin -->

## 3. Trois zones, deux fenêtres

Le cahier des charges fond deux courbes à la fois ; pour trois zones, avec $w_1$ la fenêtre de $\omega_1$ et $w_2$ celle de $\omega_2$ :

$$\mu(x) \;=\; w_1\,\mu_{\mathrm{I}}(x) \;+\; (1 - w_1)\Big[\,w_2\,\mu_{\mathrm{II}}(x) \;+\; (1 - w_2)\,\mu_{\mathrm{III}}(x)\Big].$$

Les poids des zones valent donc $w_1$, $(1-w_1)\,w_2$ et $(1-w_1)(1-w_2)$ : ils somment à 1 à tout âge, **même si les fenêtres se chevauchent** (σ grand, zone II courte : l'application prévient quand $\omega_2 - \omega_1 < 4\sigma$, mais rien ne casse).

## 4. Hors de l'étendue de chaque zone

Une zone n'est ajustée que sur ses âges, mais une fenêtre déborde de part et d'autre de la frontière. Deux règles :

- **Là où un taux lissé est stocké, on l'utilise tel quel.** En dehors des fenêtres, une seule zone a un poids de 1 et la table reprend son `mu_hat` **à l'identique**, bit pour bit.
- **Ailleurs, on évalue la formule** de la méthode à partir de ses paramètres (`evaluate_curve`) : la zone I quelques âges au-delà de $\omega_1$, la zone II des deux côtés de son étendue, la zone III sous $\omega_2$. C'est pour cela que chaque méthode a sa formule dans `joiner.evaluate_curve`, vérifiée contre le `mu_hat` de la méthode.
- Une courbe n'est évaluée **que là où son poids est positif** : un polynôme qui diverge loin de sa zone ne peut pas empoisonner la table par un produit $0 \times \infty$.

## 5. Détails

- Les âges avant le premier âge de la zone I (table qui ne commence pas à 0) sont `NaN`.
- Les erreurs possibles (messages en français) : un résultat qui n'est pas de la zone où on le place (la spline globale, de zone « tous âges », est acceptée à toute place), des frontières qui ne vérifient pas $0 \le \omega_1 < \omega_2 < \omega_3$, $\sigma \le 0$, une méthode sans formule, une zone I sans aucun âge.
- Pour une zone III qui s'arrête avant $\omega_3$, le résultat couvre déjà jusqu'à $\omega_3$ (extrapolation, voir la fiche de Kannisto).

## Références

- Cahier des charges, section 4 (raccord par moyenne pondérée en tangente hyperbolique).
