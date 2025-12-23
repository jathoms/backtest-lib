from __future__ import annotations

from datetime import date

project = "backtest-lib"
author = "jathoms"
copyright = f"{date.today().year}, {author}"
release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

myst_enable_extensions = [
    "colon_fence",
]

myst_heading_anchors = 3

doctest_global_setup = r"""
import backtest_lib
"""

doctest_default_flags = (
    1  # ELLIPSIS: allow ... in expected output
    | 2  # NORMALIZE_WHITESPACE: ignore whitespace-only differences
)

html_theme = "furo"
