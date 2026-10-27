# Spline globale (toutes les zones)

> **En une phrase.** Une seule spline cubique naturelle ajustée **une fois** sur tous les âges, de 0 à $\omega_3$, puis proposée comme candidate dans les trois zones : choisie partout, elle donne une table lissée sans raccord ; choisie dans une zone, elle est raccordée aux voisines comme n'importe quelle autre méthode.

| | |
|---|---|
| **Zone** | « Tous âges » : de l'âge 0 à $\omega_3$, indépendamment de $\omega_1$ et $\omega_2$ |
| **Modèle** | $\ln\mu_x = S(x)$, spline cubique naturelle, un nœud tous les 5 ans |
| **Estimation** | celle de la spline de la zone I, sur tous les âges |
| **Paramètres** | $K$ nœuds au total ; dans chaque zone : les nœuds qui y tombent |
| **Option** | décochée par défaut (page Paramétrage) |
| **Code** | `src/spline_all.py`, `fit_spline_all` ; `src/config.py`, `zone_view` |

## 1. Pourquoi une méthode globale

Les autres méthodes décrivent chacune une zone et sont ensuite recollées. La spline globale offre l'alternative inverse : **une** courbe souple sur toute la vie, sans frontière, que l'on compare, zone par zone, aux lois paramétriques.

## 2. L'estimation

C'est exactement `fit_spline` de la zone I (voir la fiche de la spline naturelle), appliquée à **toutes** les lignes de 0 à $\omega_3$ : mêmes nœuds (un tous les 5 ans à partir du premier âge, plus le dernier âge), mêmes poids $D_x$, même base cardinale, même prolongement par la tangente. Seuls changent l'étendue, la clé de la méthode (`spline_all`) et sa zone (`all`).

Sur la France 2020, avec les âges de 0 à 109 : 23 nœuds (0, 5, …, 105 et 109). Les âges sans décès sont écartés des nœuds, comme en zone I ; une table de moins de 3 nœuds utilisables est refusée (« table trop courte »).

## 3. Une méthode, trois lignes d'évaluation

La spline globale est évaluée **dans chaque zone**, sur les âges de données de cette zone, avec les mêmes critères que les autres méthodes. Elle apparaît donc avec une ligne par zone dans le tableau de la page Évaluation.

**Combien de paramètres compte l'AIC ?** Pas les 23 : une zone ne doit pas payer pour des nœuds situés dans une autre. `config.zone_view` donne à chaque zone **les nœuds qui tombent dans ses bornes** (bornes comprises : un nœud sur une frontière compte dans les deux zones ; au moins un). Sur la France 2020 avec les bornes par défaut : 5 nœuds en zone I, 14 en zone II, 6 en zone III.

C'est un choix de ce programme, pas du cahier des charges : il rend la comparaison avec les autres méthodes de la zone équitable sans faire payer à une zone des paramètres qui n'y servent pas. Il compte les nœuds de la zone et non ceux qui l'influencent par couplage de la spline.

## 4. Dans le raccord

- Choisie dans les **trois** zones, la spline globale est la même courbe aux trois places ; comme les poids du raccord somment à 1 à tout âge, la table finale **est** cette courbe (à $10^{-16}$ près).
- Choisie dans **une seule** zone, elle est raccordée à la zone voisine par la fenêtre en tangente hyperbolique habituelle.
- Une valeur de $\mu$ au-delà de la dernière ligne de données (110 ans quand la table s'arrête à 109) est évaluée par la formule de la spline (droite tangente).

## 5. Forces et limites

**Forces.** Aucun raccord ; très souple, donc capable de suivre la forme réelle sans loi imposée ; sans paramètre à régler ; la suggestion de l'évaluateur peut la retenir là où elle bat les lois.

**Limites.**

- **Elle gagne facilement à l'AIC sur des données nationales** : sur la France 2020 elle est la meilleure des zones II et III pour les deux sexes (voir le tableau), parce qu'avec des expositions énormes une courbe souple bat toute loi à 2 ou 3 paramètres. Un bon ajustement n'est pas un bon modèle : elle épouse aussi le bruit, et son extrapolation au-delà de la dernière ligne de données n'est qu'une droite en logarithme, sans contenu démographique.
- Plus de nœuds ($K = 23$) que de paramètres des lois : elle n'est pertinente que pour des tables assez riches.
- Elle est **décochée par défaut** : cochée, elle déplace la suggestion de la page Évaluation vers elle dès qu'elle est la meilleure.

## 6. Résultats sur la France 2020

Réglages par défaut, spline globale cochée.

<!-- resultats:debut -->
- Féminin : 23 nœuds (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 109).
- Masculin : 23 nœuds (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 109).

| Sexe | Zone | p (nœuds de la zone) | n | Log-vraisemblance | AIC | ΔAIC (zone) | p(runs) | Suggérée |
|---|---|---|---|---|---|---|---|---|
| Féminin | Zone I | 5 | 21 | -218.7 | 447.5 | 3.4 | 0.133 | non |
| Féminin | Zone II | 14 | 66 | -306.0 | 639.9 | 0.0 | 0.457 | oui |
| Féminin | Zone III | 6 | 25 | -250.3 | 512.5 | 0.0 | 0.844 | oui |
| Masculin | Zone I | 5 | 21 | -329.2 | 668.5 | 40.4 | 0.014 | non |
| Masculin | Zone II | 14 | 66 | -335.7 | 699.4 | 0.0 | 0.946 | oui |
| Masculin | Zone III | 6 | 25 | -122.9 | 257.9 | 0.0 | 0.584 | oui |
<!-- resultats:fin -->

## Références

- de Boor, C. (2001). *A Practical Guide to Splines*, éd. révisée. Springer.
- Green, P.J., Silverman, B.W. (1994). *Nonparametric Regression and Generalized Linear Models*. Chapman & Hall.
