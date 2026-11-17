# De la force de mortalité à la table de vie

> **En une phrase.** La courbe $\mu_x$ raccordée devient une table de vie complète (probabilités de décès, survivants, décès, espérance de vie) par les formules classiques, avec une racine de 100 000 et une queue ouverte à force constante après $\omega_3$.

| | |
|---|---|
| **Entrée** | $\mu_x$ pour $x = 0\ldots\omega_3$ (la sortie du raccord) |
| **Sortie** | un tableau `age, mu, qx, px, lx, dx, ex` |
| **Hypothèses** | force de mortalité constante dans l'année ; décès répartis au milieu de l'année ($a_x = \tfrac12$) ; queue ouverte à force constante |
| **Code** | `src/joiner.py` : `mu_to_q`, `build_life_table` |

## 1. Les formules

Pour chaque âge $x$ :

$$q_x = 1 - e^{-\mu_x}, \qquad p_x = e^{-\mu_x} = 1 - q_x .$$

Avec la racine $l_0 = 100\,000$ :

$$l_{x+1} = l_x\,p_x, \qquad d_x = l_x\,q_x, \qquad L_x = l_{x+1} + \tfrac12\,d_x, \qquad T_x = \sum_{y \ge x} L_y, \qquad e_x = \frac{T_x}{l_x}.$$

- $l_x$ : survivants à l'âge exact $x$ ; $d_x$ : décès entre $x$ et $x+1$.
- $L_x$ : années-personnes vécues entre $x$ et $x+1$ ; les décès de l'année sont supposés en moyenne au milieu ($a_x = \tfrac12$).
- $e_x$ : espérance de vie **complète** à l'âge exact $x$, comme l'$e_x$ de la HMD.

**Le pont $q = 1 - e^{-\mu}$.** Cette formule est celle de la **force constante** sur l'année, et non celle d'une répartition uniforme des décès (UDD). C'est elle qui est utilisée, ici et dans le raccord, de sorte qu'un $q$ se retrouve exactement à partir de $\mu$ ($\mu = -\ln(1 - q)$).

## 2. La table n'est pas refermée de force

La table s'arrête au dernier âge de $\mu$ **sans forcer $q = 1$** à $\omega_3$ : à cet âge $q$ vaut 0,4 à 0,6 sur une table réelle, et cette valeur serait écrasée. Il reste donc $l_{\omega_3}\,p_{\omega_3}$ personnes en vie à la fin du tableau. Ces survivantes ne sont pas retirées de l'espérance de vie : avec la **queue ouverte** (par défaut), elles vivent à la force constante $\mu_{\omega_3}$, ce qui ajoute $l_{\omega_3+1}/\mu_{\omega_3}$ années-personnes au dernier $L$ :

$$e_{\omega_3} \;=\; \frac{1 + p}{2} \;+\; \frac{p}{\mu_{\omega_3}}, \qquad p = e^{-\mu_{\omega_3}} .$$

Avec `open_tail = False`, seule l'année qu'elles viennent d'achever est comptée, $e_{\omega_3} = (1 + p)/2$ (la troncature finie). Conséquence : la somme des $d_x$ vaut $l_0 - l_{\omega_3+1}$ et non $l_0$.

## 3. Détails et garde-fous

- Les `NaN` en tête de $\mu$ (une table qui ne commence pas à 0) sont retirés : la table commence au premier âge disponible.
- Sont refusés, avec un message en français : aucun taux fini, un trou ou une valeur infinie après le premier âge, un taux négatif, une racine non strictement positive.
- Le tableau rendu a les colonnes `age, mu, qx, px, lx, dx, ex`, de 0 à $\omega_3$ (111 lignes par défaut).

## 4. Exemple : la France 2020

La table construite avec la sélection **suggérée** par l'évaluateur (sans spline globale) :

<!-- resultats:debut -->
| Sexe | Méthodes (suggestion de l'évaluateur) | e₀ | e₆₅ | e₈₅ | q₁₀₀ |
|---|---|---|---|---|---|
| Féminin | Polynôme deg 3 · Makeham 2e ordre · Kannisto | 85.14 | 22.92 | 7.58 | 0.3338 |
| Masculin | Polynôme deg 3 · Makeham 2e ordre · Beard | 79.12 | 18.87 | 5.95 | 0.4120 |
<!-- resultats:fin -->

Les espérances de vie à la naissance sont du même ordre que celles de la HMD pour la France 2020 (environ 85,2 ans pour les femmes et 79,2 pour les hommes).

## Références

- Chiang, C.L. (1984). *The Life Table and its Applications*. Krieger.
- Wilmoth, J.R., et al. (2021). *Methods Protocol for the Human Mortality Database*, v6. University of California, Berkeley, and Max Planck Institute for Demographic Research.
