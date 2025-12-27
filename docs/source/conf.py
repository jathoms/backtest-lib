from __future__ import annotations

import doctest
from datetime import date

doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


project = "backtest-lib"
author = "jathoms"
copyright = f"{date.today().year}, {author}"
release = "0.0.0"

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "polars": ("https://docs.pola.rs/py-polars/html/", None),
}

autodoc_type_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
    "pl.DataFrame": "polars.DataFrame",
    "pl.LazyFrame": "polars.LazyFrame",
}

napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_type_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
    "pl.DataFrame": "polars.DataFrame",
    "pl.LazyFrame": "polars.LazyFrame",
}

nitpick_ignore = [
    ("py:class", "pd.DataFrame"),
    ("py:class", "pl.DataFrame"),
    ("py:class", "pl.LazyFrame"),
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False

templates_path = ["_templates"]

autosummary_generate = True

doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

html_theme = "pydata_sphinx_theme"
