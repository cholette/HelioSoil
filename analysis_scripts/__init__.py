"""HelioSoil analysis workflows.

This package holds the shared analysis kernel (``model_pipeline``) and the three
workflow modules (``simulate``, ``model_selection``, ``experiment``),
driven by the unified command-line interface in ``cli``. Run from the repository
root:

    python -m analysis_scripts.cli simulate   --help
    python -m analysis_scripts.cli select     --help
    python -m analysis_scripts.cli experiment --help
"""
