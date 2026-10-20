"""Constants shared by the pages: Risk Data Lab brand colours, labels, the loader's vocabulary.

The brand palette is the one of `eiopa_life_charts/brand.py` (validated there): the logo navy is ink and chrome, never a data
fill; `BLUE` and `GREEN` are the data slots, used here for the sexes (Female blue, Male green).
"""
from __future__ import annotations

# --- Risk Data Lab palette -------------------------------------------------
NAVY = "#134A7C"           # logo wordmark "Risk", header bands
NAVY_DEEP = "#0E3A63"      # sidebar
NAVY_NIGHT = "#0B2A45"     # hero band of the home page
BLUE = "#1B6CA8"           # data slot 1, primary buttons
SKY = "#35A8E0"            # logo accent dot
GREEN = "#3EB553"          # data slot 2
GREEN_DEEP = "#2E9E45"     # logo wordmark "Data Lab" on white
LIME = "#7DC242"
INK = "#12283D"
INK_SOFT = "#5A6B7C"
MIST = "#F4F7FA"           # page background
RULE = "#D5DCE3"

ACCEPTED_TYPES = ["txt", "csv", "tsv", "xlsx", "xls"]  # spec 1.3 (+ tsv: Eurostat bulk files)
MODE_C = "C — Un fichier de données élémentaires"
MODE_A = "A — Taux (ou décès) + exposition, sexes en colonnes"
MODE_B = "B — Un fichier par sexe"
MODES = [MODE_C, MODE_A, MODE_B]                      # C first: the recommended pipeline
SEX_LABEL = {"Female": "Féminin", "Male": "Masculin", "both": "Les deux"}
SEX_BY_LABEL = {v: k for k, v in SEX_LABEL.items()}
SEX_COLOR = {"Female": (BLUE, "circle"), "Male": (GREEN, "square")}
VIEW_LABEL = {"period": "Tables du moment (année)", "cohort": "Tables générationnelles (génération)"}
ROLE_TEXT = {"age": "Colonne d'âge", "dx": "Colonne Dx (décès)",
             "ex": "Colonne Ex (exposition)", "mx": "Colonne mx (taux)"}
QUANTITY_ROLE = {"Dx": "dx", "Ex": "ex", "mx": "mx"}
LEVEL_TEXT = {"blocking": "Erreur bloquante", "red": "Alerte rouge", "orange": "Warning"}
FIT_STATUS_COLUMNS = {"sex": "Sexe", "zone": "Zone", "label": "Méthode", "status": "Statut",
                      "converged": "Convergence", "loglik": "Log-vraisemblance",
                      "n_iter": "Itérations", "detail": "Détail"}
