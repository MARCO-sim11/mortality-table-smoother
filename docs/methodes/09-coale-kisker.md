# Coale-Kisker (zone III)

> **En une phrase.** Pas d'estimation : on trace une droite, en logarithme de $q$, entre le taux **observé** à l'âge $\omega_2$ et une cible conventionnelle ($q = 0{,}6$ à $\omega_3$). Une formule directe, un seul paramètre, aucune itération.

| | |
|---|---|
| **Zone** | III : de $\omega_2$ à $\omega_3$ |
| **Modèle** | $q_x = q_{\omega_2}\,e^{k\,(x - \omega_2)}$, avec $q_{\omega_3} = 0{,}6$ |
| **Estimation** | aucune : formule directe, déterministe |
| **Paramètres** | 1 ($k$, `n_params = 1`) |
| **Ancrage** | le taux brut de l'âge $\omega_2$, une seule ligne |
| **Code** | `src/zone3.py`, fonction `fit_coale_kisker` |

## 1. L'idée

Coale et Kisker (1990) observent que, aux âges très élevés, les données sont si peu fiables (déclarations d'âge, effectifs minuscules) qu'il vaut mieux **imposer** une forme prudente que l'estimer. Ils prolongent $q_x$ avec un **taux de croissance exponentiel**, à partir du dernier âge bien observé, jusqu'à une valeur cible à l'âge limite.

## 2. La formule

Trois lignes, sans optimisation :

$$q_{\omega_2} = 1 - e^{-D_{\omega_2}/E_{\omega_2}}, \qquad k = \frac{\ln q_{\text{cible}} - \ln q_{\omega_2}}{\omega_3 - \omega_2}, \qquad q_x = q_{\omega_2}\,e^{k\,(x - \omega_2)} .$$

- $q_{\omega_2}$ vient du **taux brut** observé à $\omega_2$, converti en probabilité par $q = 1 - e^{-m}$, exactement comme le raccord le fait ensuite.
- $q_{\text{cible}} = 0{,}6$ est la valeur conventionnelle du cahier des charges, atteinte à $\omega_3$ ; elle n'est pas un réglage de l'application (l'argument `q_target` de la fonction permet de l'explorer).
- $q_x$ est une **droite en $\ln q$** de l'ancrage à la cible.

La courbe de force de mortalité rendue est celle qui redonne exactement ce $q$ par le pont à force constante $q = 1 - e^{-\mu}$ :

$$\mu_x = -\ln(1 - q_x) .$$

(avec $q_x$ plafonné à $1 - 10^{-15}$). Au point d'ancrage, $\hat\mu_{\omega_2}$ égale le taux brut de $\omega_2$ ; au-delà, c'est la seule extrapolation.

## 3. Ce que cela implique

- **Le taux de l'âge $\omega_2$ pilote toute la courbe** : une seule ligne brute donne l'ancrage ; pour une petite population il est bruité, et toute la courbe bouge avec lui.
- **Les lignes après $\omega_2$ ne jouent aucun rôle dans la courbe.** Elles servent uniquement à calculer `loglik`, afin que l'évaluateur puisse mettre cette courbe à côté des courbes ajustées. C'est la log-vraisemblance de Poisson d'une courbe qui n'a pas été ajustée à ces lignes, et `n_params = 1` compte $k$.
- **Exigences** : la ligne de l'âge exact $\omega_2$ doit exister, avec $E > 0$ et $D > 0$ ($q = 0$ n'a pas de logarithme) ; $q_{\text{cible}}$ doit être strictement entre 0 et 1.
- `converged = True`, `n_iter = 0`. **Avertissement** si l'ancrage atteint déjà la cible ($k \le 0$ : l'extrapolation ne croît pas).
- **Paramètres rendus** : `k`, `q_omega2`, `q_target`, `x0` ($= \omega_2$, pour que la courbe puisse être évaluée à partir des paramètres seuls) et `n_params = 1`.

## 4. Forces et limites

**Forces.** Aucune optimisation, donc aucun échec numérique ; très prudente par construction ; un seul paramètre, interprétable ; utile comme **courbe de référence** quand les données sont trop pauvres pour estimer quoi que ce soit.

**Limites.**

- **La cible de 0,6 est très inférieure à ce que montrent les données de la France 2020** : le $q$ observé y atteint déjà 0,63 à 0,86 à 109 ans. La méthode s'ajuste donc très mal à ces tables (c'est de loin la plus mauvaise à l'AIC et la part de résidus hors de $\pm 2$ est de 84 à 88 %). Rien ne l'empêche de sous-estimer fortement la mortalité d'une population qui vieillit bien.
- **Un seul âge d'ancrage** : c'est la fragilité principale.
- Elle ne fait **aucun usage** des décès observés après $\omega_2$.
- Le test des séquences n'est pas défini sur ses résidus (tous de même signe, la courbe étant systématiquement trop basse) : la règle de sélection ne peut donc pas la proposer.

## 5. Résultats sur la France 2020

Réglages par défaut (zone III : 85 à 110 ans ; la log-vraisemblance est évaluée sur les 25 lignes de données).

<!-- resultats:debut -->
| Sexe | Paramètres estimés | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | \|r\| > 2 |
|---|---|---|---|---|---|---|---|
| Féminin | k = 0.0952, q(ω₂) = 0.05553, q cible = 0.6 | 25 | -8650.0 | 17302.0 | 16690.2 | — | 88.0 % |
| Masculin | k = 0.07634, q(ω₂) = 0.08898, q cible = 0.6 | 25 | -3863.6 | 7729.2 | 7468.7 | — | 84.0 % |
<!-- resultats:fin -->

## Références

- Coale, A., Kisker, E. (1990). Defects in data on old-age mortality in the United States: new procedures for calculating mortality schedules and life tables at the highest ages. *Asian and Pacific Population Forum*, 4(1).
