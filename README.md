# mortality-table-smoother

> Lissage paramétrique de tables de mortalité brutes (HMD, CSV, Excel) par zone d'âge,
> sans package actuariel — implémenté from scratch en Python avec interface Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/Licence-MIT-green)

---

## À propos

Cet outil permet de charger des données de mortalité brutes, de paramétrer et comparer
plusieurs méthodes de lissage par tranche d'âge, d'évaluer les résultats selon des
critères statistiques standards (AIC, BIC, χ², runs test), puis d'exporter la table lissée finale.

Toute la mécanique d'optimisation est codée **from scratch** — Newton-Raphson, profil de
vraisemblance, Nelder-Mead, grid search — sans dépendance à un package actuariel.

> **Référence méthodologique** : les méthodes implémentées suivent Gompertz (1825),
> Makeham (1860), Kannisto (1994), Coale & Kisker (1990), Richards (2012).
> Document de référence complet disponible dans `docs/lissage_mortalite_v1.1.docx`.

---

## Décomposition par zone d'âge

| Zone    | Plage      | Méthodes disponibles                              | Estimation                          |
|---------|------------|---------------------------------------------------|-------------------------------------|
| Zone I  | 0 – ω₁     | Polynôme deg 3, Splines cubiques naturelles       | WLS — â = (AᵀWA)⁻¹·Aᵀ·W·y         |
| Zone II | ω₁ – ω₂   | Gompertz, Makeham, Makeham 2ᵉ ordre, Perks        | MLE Poisson — Newton-Raphson / NM   |
| Zone III| ω₂ – ω₃   | Kannisto, Beard, Coale-Kisker                     | Grid search 2D / NR / formule directe |

La jonction entre zones est réalisée par interpolation `tanh` sur une fenêtre glissante
[ωᵢ ± 2σ] pour éviter les discontinuités.

Conversion finale : `qₓ = 1 − exp(−μₓ)` (hypothèse UDD).

---

## Structure du projet

```
mortality-table-smoother/
├── src/
│   ├── loader.py          # Chargement HMD .txt / CSV / Excel
│   ├── config.py          # SmootherConfig — zones, méthodes, seuils
│   ├── zone1_poly.py      # Polynôme deg 3, splines naturelles
│   ├── zone2_gompertz.py  # Gompertz — Newton-Raphson 2×2
│   ├── zone2_makeham.py   # Makeham — NR 3×3 + profil sur c
│   ├── zone2_perks.py     # Perks, Makeham 2ᵉ ordre — Nelder-Mead
│   ├── zone3.py           # Kannisto, Beard, Coale-Kisker
│   ├── joiner.py          # Raccord inter-zones (tanh)
│   ├── evaluator.py       # χ², AIC/BIC, runs test, résidus déviance
│   └── exporter.py        # Export Excel / CSV / TXT (format HMD)
├── app.py                 # Interface Streamlit — 4 onglets
├── docs/
│   ├── lissage_mortalite_v1.1.docx   # Référence méthodologique
│   └── mortality_smoother_specs.docx # Cahier des charges technique
├── data/                  # Données HMD (non versionnées — voir .gitignore)
├── tests/
│   └── test_*.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<username>/mortality-table-smoother.git
cd mortality-table-smoother
pip install -r requirements.txt
streamlit run app.py
```

### Dépendances

```
numpy
scipy
pandas
openpyxl
streamlit
matplotlib
plotly
```

Aucun package actuariel externe. SciPy est utilisé uniquement pour
`scipy.optimize.minimize` (Nelder-Mead) et `scipy.interpolate.CubicSpline`.

---

## Données d'entrée

| Format               | Extension     | Notes                                                       |
|----------------------|---------------|-------------------------------------------------------------|
| HMD Deaths           | `.txt`        | Entête `#`, séparateur espace — colonnes Year Age Female Male Total |
| HMD Exposure-to-risk | `.txt`        | Même format. Nécessaire pour le MLE (Dₓ et Eₓ séparés)    |
| CSV générique        | `.csv`        | Séparateur `,` ou `;`, mapping colonnes configurable        |
| Excel                | `.xlsx / .xls`| Feuille sélectionnable, entête configurable                 |

Les données HMD sont téléchargeables sur [mortality.org](https://www.mortality.org)
(accès gratuit sur inscription). Fichiers requis : **Deaths 1×1** + **Exposure-to-risk 1×1**.

---

## Export

- **Excel (.xlsx)** — 4 feuilles : table lissée, taux bruts, paramètres estimés, critères de sélection
- **CSV** — colonnes `age, mu, qx, px, lx, dx, ex`, séparateur configurable
- **TXT** — format HMD (entête standardisé, séparateur espace)

---

## Interface Streamlit — 4 onglets

| # | Onglet                        | Contenu                                                                          |
|---|-------------------------------|----------------------------------------------------------------------------------|
| 1 | Chargement & Contrôle         | Upload, mapping colonnes, contrôles qualité, graphique log(mₓ) bruts           |
| 2 | Paramétrage & Lancement       | Sliders ω₁/ω₂, checkboxes méthodes par zone, σ de jonction, bouton "Lancer"    |
| 3 | Évaluation & Sélection        | Tableau AIC/BIC/χ²/runs test, résidus de déviance, sélection méthode finale     |
| 4 | Visualisation & Export        | Courbe lissée vs taux bruts, prévisualisation table qₓ, boutons d'export        |

---

## Références

- Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. *Phil. Trans. R. Soc.*
- Makeham, W.M. (1860). On the law of mortality. *J. Inst. Actuaries.*
- Kannisto, V. (1994). *Development of oldest-old mortality.* Odense Monographs on Population Aging.
- Coale, A., Kisker, E. (1990). Defects in data on old-age mortality. *European Journal of Population.*
- Richards, S.J. (2012). A handbook of parametric survival models for actuarial use. *Scandinavian Actuarial Journal.*
- Human Mortality Database — [mortality.org](https://www.mortality.org)

---

## Licence

MIT — libre d'utilisation, de modification et de distribution.

---

*Marc Jodel Simo — 2026*
