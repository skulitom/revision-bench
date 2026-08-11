"""Per-passage, per-round readouts (plan.md §7).

Everything in this package is pure Python plus numpy: no network, no API key, no GPU.
That is a deliberate property, not an accident of the current milestone. Phase 0's
acceptance criterion is "metrics reproducible from artifacts alone" (plan.md §9), which
only stays checkable while recomputing every number costs nothing but CPU.

Implemented so far (milestone M0-b):

- :mod:`~revisionbench.metrics.stats` — bootstrap CIs and the percentile convention.
- :mod:`~revisionbench.metrics.stylometry` — M1 identity, M2 homogenization.
- :mod:`~revisionbench.metrics.slop` — M3 slop index.
- :mod:`~revisionbench.metrics.thrash` — M6 thrash and convergence.

M4 (blinded quality / Bradley–Terry), M5 (defect-fix recall) and M7 (judge validity) are
not here, because they need judge panels or a defect manifest and belong to later
milestones. There are no placeholder modules for them on purpose: an absent module is
honest, and a stub that returns a plausible zero is the thing this repo is organised
against.

Note that "no model" is a property of this package, not of the project: the revision loop
in :mod:`revisionbench.loop` does call a model, and writes only text and generation facts
so that everything here stays a pure function of the artifact.
"""

__all__: list[str] = []
