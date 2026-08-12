"""
Evaluation suite.

Exports are lazy on purpose. metrics.py imports RAGAS, pandas and the OpenAI
async client — several seconds and a large dependency tree. Eagerly re-exporting
it here meant that `from evals.pipeline import detect_tool`, which needs none of
that, paid the full cost anyway and failed outright in any environment without
the eval extras installed.

__getattr__ defers each import until the name is actually used.
"""

__all__ = [
    "run_pipeline",
    "load_golden_dataset",
    "run_guardrails_eval",
    "compute_guardrails_metrics",
    "run_all_metrics",
]

_EXPORTS = {
    "run_pipeline": "evals.pipeline",
    "load_golden_dataset": "evals.pipeline",
    "run_guardrails_eval": "evals.guardrails_eval",
    "compute_guardrails_metrics": "evals.guardrails_eval",
    "run_all_metrics": "evals.metrics",
}


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module

        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
