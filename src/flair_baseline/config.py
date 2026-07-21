"""Small helpers: task-name resolution and the default ELF config path."""
from __future__ import annotations

from flair_benchmark.tasks import list_tasks

DEFAULT_ELF_CONFIG = "flair_elf_config.yaml"

# Tasks the baseline does not run, even though flair_benchmark still defines them.
# The sepsis task (task4) is retired here: it is out of scope for this baseline, so
# it is filtered out at the single choke point below rather than at each call site.
# Removing an entry re-enables the task; no other code change is needed.
EXCLUDED_TASKS = frozenset({"task4_sepsis_abx_6h"})


def available_tasks() -> list[str]:
    """Benchmark tasks this baseline runs — ``list_tasks()`` minus ``EXCLUDED_TASKS``.

    Single source of truth for "which tasks exist" across the CLI: every stage
    defaults to this list, and ``resolve_task`` matches ``--task`` against it, so an
    excluded task is neither run by default nor selectable by name.
    """
    return [t for t in list_tasks() if t not in EXCLUDED_TASKS]


def resolve_task(name: str) -> str:
    """Accept a full task name or a short prefix (e.g. 'task1') → full task name."""
    tasks = available_tasks()
    if name in tasks:
        return name
    matches = [t for t in tasks if t == name or t.split("_")[0] == name or t.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if name in EXCLUDED_TASKS or any(
            t == name or t.split("_")[0] == name or t.startswith(name)
            for t in EXCLUDED_TASKS
        ):
            raise ValueError(
                f"Task {name!r} has been removed from this baseline. Available: {tasks}")
        raise ValueError(f"Unknown task {name!r}. Available: {tasks}")
    raise ValueError(f"Ambiguous task {name!r} → {matches}. Use the full name.")
