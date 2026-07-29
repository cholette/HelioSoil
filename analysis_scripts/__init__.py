"""HelioSoil analysis workflows.

This package holds the shared analysis kernel (``model_pipeline``) and the three
workflow modules (``single_campaign_fit``, ``model_selection``, ``experiment``),
driven by the unified command-line interface in ``cli``. Run from the repository
root:

    python -m analysis_scripts.cli fit        --help
    python -m analysis_scripts.cli select     --help
    python -m analysis_scripts.cli experiment --help
"""
