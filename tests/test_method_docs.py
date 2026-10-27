"""docs/methodes/*.md - the method documents. Run:   python -m pytest tests/test_method_docs.py -v

A document of this kind rots silently: a constant changes in the code, a function is renamed, a table keeps its old numbers. These tests
tie the documents to the program:

  * the tables of results are the ones `docs/build_method_results.py` computes now (France 2020, default settings);
  * every method of the registry has a document, and every document is listed on the page;
  * a document only cites files, functions and constants that exist, and the numbers it states about the algorithm (grid sizes, seeds,
    caps, targets...) are the ones of the code;
  * the LaTeX is well formed and uses macros KaTeX (the renderer of Streamlit) knows.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs"))

import build_method_results as B           # noqa: E402
from src import config as C                # noqa: E402
from src import evaluator as EV            # noqa: E402
from src import fit_common as FC           # noqa: E402
from src import joiner as J                # noqa: E402
from src import zone1_poly as Z1           # noqa: E402
from src import zone2_gompertz as Z2G      # noqa: E402
from src import zone2_makeham as Z2M       # noqa: E402
from src import zone2_makeham2 as Z2M2     # noqa: E402
from src import zone2_perks as Z2P         # noqa: E402
from src import zone3 as Z3                # noqa: E402
from ui import methodology as M            # noqa: E402

DOCS = {d.slug: d for d in M.all_docs()}
SLUGS = list(DOCS)
TEXT = {slug: M.read_doc(slug) for slug in SLUGS}
KATEX_MACROS = {"Big", "Delta", "Gamma", "Longleftrightarrow", "alpha", "approx", "bar", "begin", "beta", "big", "binom", "cdot", "chi",
                "delta", "dfrac", "dots", "ell", "end", "eta", "exp", "frac", "gamma", "ge", "hat", "in", "infty", "ldots", "le",
                "leftarrow", "ll", "ln", "lvert", "mathbb", "mathcal", "mathrm", "max", "mu", "omega", "operatorname", "overline",
                "partial", "phi", "pm", "qquad", "quad", "rvert", "sigma", "sim", "sqrt", "subset", "sum", "tanh", "tau", "text",
                "tfrac", "theta", "times", "top", "underbrace"}


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def formulas(text: str) -> list:
    """Every formula of a document: display ($$...$$) and inline ($...$)."""
    text = strip_code(text)
    found = re.findall(r"\$\$(.+?)\$\$", text, flags=re.S)
    rest = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.S)
    return found + re.findall(r"\$([^$\n]+)\$", rest)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_every_method_of_the_registry_has_a_document_and_no_document_is_orphaned_or_missing():
    assert {d.key for d in M.all_docs() if d.key} == {m.key for m in C.every_method()}
    files = sorted(p.stem for p in M.DOC_DIR.glob("*.md") if p.stem != "README")
    assert sorted(SLUGS) == files and len(set(SLUGS)) == len(SLUGS)
    assert SLUGS == sorted(SLUGS)                                      # numbered in the order of the page


def test_the_index_lists_every_document():
    index = (M.DOC_DIR / "README.md").read_text(encoding="utf-8")
    for slug in SLUGS:
        assert f"({slug}.md)" in index, slug
    assert index.count("](") >= len(SLUGS)


@pytest.mark.parametrize("slug", SLUGS)
def test_a_document_has_one_title_a_one_sentence_summary_and_its_references(slug):
    lines = TEXT[slug].splitlines()
    plain = strip_code(TEXT[slug]).splitlines()
    assert lines[0].startswith("# ") and sum(1 for ln in plain if ln.startswith("# ")) == 1
    assert lines[2].startswith("> **En une phrase.**")
    assert "## Références" in TEXT[slug] and "onglet" not in TEXT[slug].lower()          # the app has pages, not tabs


@pytest.mark.parametrize("slug", [s for s in SLUGS if DOCS[s].key])
def test_a_method_document_has_its_identity_card_with_the_real_module_and_function(slug):
    spec = C.method_spec(DOCS[slug].key)
    text = TEXT[slug]
    for row in ("**Zone**", "**Modèle**", "**Estimation**", "**Paramètres**", "**Code**"):
        assert f"| {row} |" in text, row
    assert f"src/{spec.module}.py" in text and f"`{spec.fit_name}`" in text


@pytest.mark.parametrize("slug", SLUGS)
def test_a_results_block_exists_exactly_where_the_generator_makes_one(slug):
    assert (B.current_block(TEXT[slug]) is not None) == (slug in B.BLOCKS)
    assert TEXT[slug].count(B.START) == TEXT[slug].count(B.END) == (1 if slug in B.BLOCKS else 0)


# --------------------------------------------------------------------------
# The numbers are the code's
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list(B.BLOCKS))
def test_the_results_table_is_the_one_the_code_produces_now(slug):
    assert B.current_block(TEXT[slug]) == B.BLOCKS[slug](), f"{slug}.md is out of date: run python docs/build_method_results.py"


def test_the_generator_check_mode_agrees(capsys):
    assert B.main(check=True) == 0
    assert "nothing" in capsys.readouterr().out


def fr(value, digits: int | None = None) -> str:
    """A number as the documents write it in LaTeX: a decimal comma in braces."""
    text = f"{value:g}" if digits is None else f"{value:.{digits}f}"
    return text.replace(".", "{,}")


CLAIMS = [
    # (document, what it says, the code constant it must agree with)
    ("04-makeham", f"{Z2M.GRID_N} points"),
    ("04-makeham", fr(Z2M.C_BOUNDS[0])), ("04-makeham", fr(Z2M.C_BOUNDS[1])),
    ("05-makeham-2e-ordre", f"d = {fr(Z2M2.SEED_BASE_RATIO)}"),
    ("05-makeham-2e-ordre", f"**{round(100 * Z2M2.SEED_AMPLITUDE)} %**"),
    ("06-perks", f"**{round(100 * Z2P.SEED_DECELERATION)} %**"),
    ("07-kannisto", "100 × 100"), ("07-kannisto", f"[{Z3.LN_A_RANGE[0]:g}, {Z3.LN_A_RANGE[1]:g}]"),
    ("07-kannisto", fr(Z3.B_RANGE[0], 2)), ("07-kannisto", fr(Z3.B_RANGE[1], 2)), ("07-kannisto", f"plus de {Z3.TRUST_LOGIT:g}"),
    ("09-coale-kisker", fr(Z3.Q_TARGET)),
    ("02-spline-naturelle", f"tous les {Z1.KNOT_STEP} ans"), ("02-spline-naturelle", f"Minimum de {Z1.MIN_KNOTS} nœuds"),
    ("00-cadre-general", f"{FC.NM_MAXITER // 1000} {FC.NM_MAXITER % 1000:03d} itérations"),
    ("00-cadre-general", f"au plus {FC.NM_RESTARTS} fois"),
    ("00-cadre-general", f"côté {str(0.05).replace('.', ',')}"),
    ("00-cadre-general", f"{Z2G.MAX_HALVINGS} fois"),
    ("00-cadre-general", "(200 par défaut)"), ("00-cadre-general", "(1e-8 par défaut)"),
    ("11-raccord", f"$\\pm {J.WINDOW:g}\\sigma$"), ("11-raccord", "CONTINUOUS_EDGES"),
    ("11-raccord", f"{C.DEFAULT_SIGMA_JOIN:g} par défaut, de {C.SIGMA_JOIN_RANGE[0]:g} à {C.SIGMA_JOIN_RANGE[1]:g}"),
    ("12-evaluation", f"{fr(EV.RUNS_ALPHA, 2)}"), ("12-evaluation", f"{EV.WH_H:g} par défaut"),
    ("13-table-de-vie", f"l_0 = {int(J.RADIX):,}".replace(",", "\\,")),
]


@pytest.mark.parametrize("slug,snippet", CLAIMS)
def test_what_a_document_states_about_the_algorithm_is_what_the_code_does(slug, snippet):
    assert snippet in TEXT[slug], f"{slug}.md no longer says {snippet!r}: the code or the document changed"


def test_the_defaults_quoted_in_the_documents_are_the_defaults_of_the_config():
    cfg = C.SmootherConfig()
    assert (cfg.omega1, cfg.omega2, cfg.omega3, cfg.sigma_join, cfg.nr_maxiter, cfg.nr_tol, cfg.grid_n) == (20, 85, 110, 3, 200, 1e-8, 100)
    assert "20, 85 et 110" in TEXT["00-cadre-general"]


# --------------------------------------------------------------------------
# What a document cites exists
# --------------------------------------------------------------------------

SRC_MODULES = sorted(p.stem for p in (ROOT / "src").glob("*.py") if p.stem != "__init__")
TOKENS = {slug: re.findall(r"`([^`\n]+)`", strip_code(text)) for slug, text in TEXT.items()}


def namespace(name: str):
    return importlib.import_module(f"src.{name}")


@pytest.mark.parametrize("slug", SLUGS)
def test_a_document_cites_only_files_functions_and_constants_that_exist(slug):
    fit_names = {m.fit_name for m in C.every_method()}
    constants = {n for mod in SRC_MODULES for n in dir(namespace(mod)) if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", n)}
    for token in TOKENS[slug]:
        if re.fullmatch(r"(src|docs|tests|sample_data)/[\w/.\-]+", token):
            assert (ROOT / token).exists(), f"{slug}: {token} does not exist"
        elif re.fullmatch(r"fit_\w+", token):
            assert token in fit_names or token in SRC_MODULES, f"{slug}: {token} is not a fit function nor a module"
        elif re.fullmatch(r"(\w+)\.(\w+)", token) and token.split(".")[0] in SRC_MODULES:
            module, attr = token.split(".")
            assert hasattr(namespace(module), attr), f"{slug}: {token} does not exist"
        elif re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", token):
            assert token in constants, f"{slug}: constant {token} does not exist in src/"
        elif token in ("nr_tol", "nr_maxiter", "grid_n"):
            assert hasattr(C.SmootherConfig(), token), f"{slug}: {token} is not a setting"


# --------------------------------------------------------------------------
# The LaTeX
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_the_dollar_signs_pair_up_and_every_formula_is_balanced(slug):
    text = strip_code(TEXT[slug])
    assert text.count("$$") % 2 == 0
    rest = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.S)
    assert rest.count("$") % 2 == 0, "an unpaired $ : a stray dollar sign or an unclosed inline formula"
    for formula in formulas(TEXT[slug]):
        bare = re.sub(r"\\[{}]", "", formula)
        assert bare.count("{") == bare.count("}"), formula
        assert len(re.findall(r"\\left(?![A-Za-z])", formula)) == len(re.findall(r"\\right(?![A-Za-z])", formula)), formula
        assert len(re.findall(r"\\begin\{", formula)) == len(re.findall(r"\\end\{", formula)), formula


@pytest.mark.parametrize("slug", SLUGS)
def test_a_document_only_uses_latex_macros_katex_knows(slug):
    used = {m for formula in formulas(TEXT[slug]) for m in re.findall(r"\\([A-Za-z]+)", formula)}
    assert used <= KATEX_MACROS, sorted(used - KATEX_MACROS)


def test_the_documents_describe_formulas_at_all():
    assert sum(len(formulas(t)) for t in TEXT.values()) > 150
