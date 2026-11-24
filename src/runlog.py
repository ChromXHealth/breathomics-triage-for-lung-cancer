"""Lightweight run logging for workflow entry points.

Provides two helpers:

- ``step(index, total, title)`` prints a step header such as ``[3/5] Sensitivity Correction``.
- ``note(message)`` prints an indented bullet for sub-conclusions within a step.

Workflow scripts own the step numbering; this module only formats output.
"""

from __future__ import annotations


def step(index: int, total: int, title: str) -> None:
    """Print a step header line, e.g. ``[3/5] Sensitivity Correction``."""
    print(f"\n[{index}/{total}] {title}", flush=True)


def note(message: str) -> None:
    """Print an indented sub-conclusion within the current step."""
    print(f"  - {message}", flush=True)


def final_outputs(entries: list[tuple[str, str]]) -> None:
    """Print the key output files and the role each one plays downstream.

    ``entries`` is a list of ``(path, purpose)`` pairs; paths are printed
    exactly as supplied (callers pass their output-dir-relative paths).
    """
    print("\nFinal outputs (for downstream use):", flush=True)
    for path, purpose in entries:
        print(f"  - {path}: {purpose}", flush=True)
