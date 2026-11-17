# Les méthodes du lisseur

Une fiche par méthode, dans l'ordre où l'application les propose, plus quatre fiches sur ce qui les entoure. Les mêmes textes sont affichés dans la page **Méthodologie** de l'application.

Chaque fiche suit le même plan : fiche d'identité, le modèle, l'estimation pas à pas, les décisions d'implémentation, les cas limites, forces et limites, **résultats sur la France 2020** (tableau produit par le code), références.

| # | Fiche | Zone | Modèle |
|---|---|---|---|
| 00 | [Cadre général](00-cadre-general.md) | toutes | notations, vraisemblance de Poisson, estimateurs |
| 01 | [Polynôme de degré 3](01-polynome-degre-3.md) | I | $\ln\mu_x$ polynomial, moindres carrés pondérés |
| 02 | [Spline cubique naturelle](02-spline-naturelle.md) | I | $\ln\mu_x$ spline, nœuds tous les 5 ans |
| 03 | [Gompertz](03-gompertz.md) | II | $\mu_x = B\,c^x$ |
| 04 | [Makeham](04-makeham.md) | II | $\mu_x = A + B\,c^x$ |
| 05 | [Makeham du second ordre](05-makeham-2e-ordre.md) | II | $\mu_x = A + B\,c^x + D\,d^x$ |
| 06 | [Perks](06-perks.md) | II | $\mu_x = (A + B\,c^x)/(1 + D\,c^x)$ |
| 07 | [Kannisto](07-kannisto.md) | III | logistique, plateau 1 |
| 08 | [Beard](08-beard.md) | III | $\mu_x = B\,c^x/(1 + K\,B\,c^x)$, plateau $1/K$ |
| 09 | [Spline globale](09-spline-globale.md) | toutes | une spline sur $0\ldots\omega_3$ |
| 10 | [Le raccord des zones](10-raccord.md) | — | pondération en tangente hyperbolique |
| 11 | [Évaluation et sélection](11-evaluation.md) | — | AIC, BIC, $\chi^2$, séquences, Whittaker-Henderson |
| 12 | [Table de vie](12-table-de-vie.md) | — | $q_x$, $l_x$, $d_x$, $e_x$ |

## Entretenir ces documents

Les tableaux « Résultats sur la France 2020 » sont **générés** (entre les balises `<!-- resultats:debut -->` et `<!-- resultats:fin -->`) :

```bash
python docs/build_method_results.py          # réécrit les tableaux
python docs/build_method_results.py --check  # dit lesquels sont périmés
```

Des tests, non publiés dans ce dépôt, échouent si un tableau ne correspond plus au code, si une fiche cite un fichier ou une fonction qui n'existe pas, ou si une méthode du registre n'a pas de fiche.
