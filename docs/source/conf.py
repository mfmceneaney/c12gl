# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Add source files for autodoc
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)+"/../../c12gl"))

# -- Project information

project = 'c12gl'
copyright = '2022, Matthew McEneaney'
author = 'Matthew McEneaney'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
