"""Form analysis: prepared statistics plus an agent that reads them.

Typical use::

    from forms_ai import analyze_form

    result = await analyze_form(
        "5ad8596a-077f-4bbe-abdf-b42cb5bfadbb",
        period_from="2025-01-01", period_to="2025-03-31",
        representative_id="5ad8596a-077f-4bbe-abdf-b42cb5bfadbb",
        granularity="auto",
        analysis_rules={"audience": "regional sales manager",
                        "thresholds": {"yes_rate_floor": 0.8}},
    )
    print(result.report_markdown)
"""

from .core.prompts import AnalysisRules
from .forms.analysis import (FormAnalysisResult, analyze_form, analyze_form_json,
                             list_representatives)
from .forms.filters import FilterError, FilterSpec
from .forms.loader import DatasetNotFound, load_dataset
from .forms.schema import SchemaError

__all__ = [
    "analyze_form", "analyze_form_json", "list_representatives",
    "FormAnalysisResult", "AnalysisRules", "FilterSpec",
    "load_dataset", "DatasetNotFound", "SchemaError", "FilterError",
]
