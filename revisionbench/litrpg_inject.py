"""Plant contradictions in a synthetic LitRPG manuscript.

Stratum B corrupts real prose and records where. This corrupts prose *rendered from a
manifest*, so the ground truth is not an annotation but the thing the text was generated
from — and a contradiction is a provable disagreement with a table rather than a judgement.

The five types are the genre's real failure modes, not a taxonomy invented to be findable:

- ``stat_drift`` — a status block reports a stat the character does not have
- ``level_regression`` — level falls or skips between chapters
- ``skill_before_acquisition`` — a skill is used chapters before it is learned
- ``inventory_ghost`` — an item is used after being lost, or before being gained
- ``entity_rename`` — a skill or item name drifts to a near-variant

Three of these are **cross-chapter**: nothing in a single chapter is wrong, and the
contradiction exists only against what an earlier or later chapter established. That is the
class `harness-gap.md` §4 says hurts a manuscript most and that no existing metric in this
repo can express, which is the reason this module exists.

Injection is per-chapter and the chapters are joined afterwards, so a defect's span never
needs to be re-derived after a later injection shifts it. `inject.py` learned that the hard
way and re-locates fragments by content for the same reason.

**This module must never be imported by a detector.** Enforced by test, as with `detect.py`
— a detector that can see how a defect was planted is measuring the injector, not the text.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from revisionbench.litrpg import Manifest, render_chapter, render_status_block
from revisionbench.litrpg_detect import KNOWN_FIELDS

__all__ = [
    "LITRPG_DEFECT_TYPES",
    "LITRPG_INJECTOR_VERSION",
    "LitRPGDefect",
    "LitRPGInjectionError",
    "inject_manuscript",
]

LITRPG_INJECTOR_VERSION = 1

LITRPG_DEFECT_TYPES = (
    "stat_drift",
    "level_regression",
    "skill_before_acquisition",
    "inventory_ghost",
    "entity_rename",
)


class LitRPGInjectionError(RuntimeError):
    """A manuscript does not admit the requested defect."""


@dataclass(frozen=True, slots=True)
class LitRPGDefect:
    """One planted contradiction.

    Attributes:
        cross_chapter: True when the corrupted chapter is internally consistent and only
            contradicts a *different* chapter. Reported separately in scoring, because
            within-chapter and cross-chapter detection are different problems and a
            recall number that pools them hides which one a detector actually solves.
        established_in: The chapter that makes it a contradiction. Empty for within-chapter
            defects.
    """

    defect_id: str
    manuscript_id: str
    type: str
    chapter: int
    original_fragment: str
    corrupt_fragment: str
    char_span: tuple[int, int]
    clean_line: str
    cross_chapter: bool
    established_in: int = 0
    injector_version: int = LITRPG_INJECTOR_VERSION
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["char_span"] = list(self.char_span)
        return data


def _variant(name: str, rng: random.Random) -> str:
    """A plausible near-miss of a proper name, as a tired author would mistype it.

    Deliberately not a random string: the defect must be the kind a reader could miss and a
    detector must earn. ``Ember Lash`` -> ``Ember Whip`` is drift; ``Ember Xqz`` is a typo
    any dictionary check would catch, and finding it would prove nothing.
    """
    swaps = {
        "Lash": "Whip",
        "Ward": "Guard",
        "Step": "Stride",
        "Spike": "Lance",
        "Skin": "Hide",
        "Nail": "Spike",
        "Sight": "Vision",
        "Ledger": "Tally",
        "Palm": "Hand",
        "Coil": "Vine",
        "Beckon": "Summons",
        "Song": "Chant",
        "whetstone": "grindstone",
        "compass": "dial",
        "vial": "phial",
        "key": "latchkey",
        "charm": "token",
        "pouch": "sack",
        "ring": "band",
        "map": "chart",
        "lantern": "lamp",
        "cloak": "mantle",
    }
    parts = name.split()
    for i, part in enumerate(parts):
        if part in swaps:
            return " ".join([*parts[:i], swaps[part], *parts[i + 1 :]])
    return name + ("s" if not name.endswith("s") else "")


def _body_start(text: str) -> int:
    """Where the narrative begins, i.e. after the last status-block field."""
    last = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        field = stripped.split(":", 1)[0] if ":" in stripped else ""
        if field in KNOWN_FIELDS:
            last = text.find(line) + len(line)
    return last


def _replace_once(
    text: str, old: str, new: str, *, in_body: bool = False
) -> tuple[str, tuple[int, int]]:
    """Replace the first occurrence; with ``in_body``, the first occurrence *in the prose*.

    ``in_body`` is not a nicety. A skill name's first occurrence in a chapter is inside the
    status block's ``Skills:`` line, so a naive replace corrupts the canonical record rather
    than the narrative — which makes the planted defect agree with itself and therefore
    undetectable by construction. That silently cost `skill_before_acquisition` and
    `entity_rename` their entire recall before this existed.
    """
    start = _body_start(text) if in_body else 0
    index = text.find(old, start)
    if index < 0:
        raise LitRPGInjectionError(f"fragment not present{' in body' if in_body else ''}: {old!r}")
    return text[:index] + new + text[index + len(old) :], (index, index + len(new))


def _line_containing(text: str, span: tuple[int, int]) -> str:
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    return text[start : end if end >= 0 else len(text)]


def inject_manuscript(
    manifest: Manifest,
    *,
    types: list[str] | None = None,
    per_type: int = 2,
    seed: int = 0,
    chapters_text: dict[int, str] | None = None,
) -> tuple[str, list[LitRPGDefect]]:
    """Render the manuscript and plant defects. Returns (corrupted_text, defects).

    Spans are in *final manuscript* coordinates: chapters are corrupted independently and
    the offsets are applied once at join time, so no span is ever invalidated by a later
    injection.

    **Fewer than ``per_type * len(types)`` defects may be planted**, and that is not an
    error: a type declines chapters it cannot corrupt honestly (a stat change in a level-up
    chapter is legal, not a defect), and only one defect goes in each chapter. The shortfall
    is reported in :attr:`LitRPGDefect.notes` on every returned defect so a caller cannot
    quote recall against a denominator that was never planted — the shortfall is invisible
    otherwise, and a run that planted 4 of 10 looks identical to one that planted 10.
    """
    chosen = list(types or LITRPG_DEFECT_TYPES)
    unknown = [t for t in chosen if t not in LITRPG_DEFECT_TYPES]
    if unknown:
        raise ValueError(f"unknown defect type(s): {unknown}; known: {list(LITRPG_DEFECT_TYPES)}")

    rng = random.Random(f"{manifest.manuscript_id}:{seed}")
    # ``chapters_text`` lets a model-written manuscript be corrupted by the same code that
    # corrupts the templated one. The manifest is ground truth either way, which is the
    # property that makes the substitution safe — see litrpg.prompt_for_chapter.
    chapters = chapters_text or {
        c.chapter: render_chapter(manifest, c.chapter) for c in manifest.chapters
    }
    # One defect per chapter at most. Two contradictions in one chapter can interact —
    # renaming a skill that a use-before-acquisition defect also moved leaves a defect whose
    # "correct" repair is ambiguous, and an ambiguous target cannot be scored.
    available = [c.chapter for c in manifest.chapters if c.chapter > 1]
    rng.shuffle(available)
    planted: list[LitRPGDefect] = []

    requested = len(chosen) * per_type

    for defect_type in chosen:
        for _ in range(per_type):
            if not available:
                break
            chapter = available.pop()
            state = manifest.state_at(chapter)
            text = chapters[chapter]

            try:
                if defect_type == "stat_drift":
                    # Change a stat in a chapter with no level-up, so the manifest's rule
                    # "stats move only on level-up" makes it a contradiction outright.
                    if state.levelled_up:
                        raise LitRPGInjectionError("level-up chapter admits a stat change")
                    stat = rng.choice(list(state.stats))
                    old = f"  {stat}: {state.stats[stat]}"
                    new = f"  {stat}: {state.stats[stat] + rng.choice([-2, -1, 1, 2])}"
                    text, span = _replace_once(text, old, new)
                    defect = (old, new, True, chapter - 1)

                elif defect_type == "level_regression":
                    # Only in a chapter with no level-up. In a level-up chapter the
                    # character's level equals the previous chapter's plus one, so
                    # decrementing it lands exactly on the previous value — neither a
                    # fall nor a skip, and no longer the defect this type names.
                    if state.levelled_up:
                        raise LitRPGInjectionError("level-up chapter: -1 ties the previous level")
                    old, new = f"  Level: {state.level}", f"  Level: {state.level - 1}"
                    text, span = _replace_once(text, old, new)
                    defect = (old, new, True, chapter - 1)

                elif defect_type == "skill_before_acquisition":
                    later = [
                        s
                        for c in manifest.chapters
                        if c.chapter > chapter
                        for s in c.skills
                        if s not in state.skills
                    ]
                    if not later:
                        raise LitRPGInjectionError("no later skill to borrow")
                    future = rng.choice(later)
                    text, span = _replace_once(text, state.skills[-1], future, in_body=True)
                    acquired = next(c.chapter for c in manifest.chapters if future in c.skills)
                    defect = (state.skills[-1], future, True, acquired)

                elif defect_type == "inventory_ghost":
                    # Exclude anything held in the neighbouring chapters. Prose may
                    # legitimately name an item in the chapter it is lost, so planting a
                    # "ghost" that was held next door plants a non-defect, and scoring it
                    # as a miss would understate the detector rather than test it.
                    nearby = set(state.inventory)
                    for neighbour in (chapter - 1, chapter + 1):
                        if 1 <= neighbour <= len(manifest.chapters):
                            nearby |= set(manifest.state_at(neighbour).inventory)
                    absent = [i for c in manifest.chapters for i in c.inventory if i not in nearby]
                    if not absent or not state.inventory:
                        raise LitRPGInjectionError("no absent item to reference")
                    ghost = rng.choice(absent)
                    held = state.inventory[0]
                    old, new = f"the {held}", f"the {ghost}"
                    text, span = _replace_once(text, old, new, in_body=True)
                    # Within-chapter: this chapter's own Inventory line already
                    # contradicts the prose, so no other chapter is needed to see it.
                    defect = (old, new, False, chapter)

                else:  # entity_rename
                    # Only entities the rendered prose actually names. Picking any
                    # canonical entity means most attempts find nothing in the body and
                    # silently plant nothing, which reads as a detector miss.
                    in_prose = [
                        state.skills[-1],
                        *([state.inventory[0]] if state.inventory else []),
                        *([state.item_gained] if state.item_gained else []),
                        *([state.item_lost] if state.item_lost else []),
                    ]
                    target = rng.choice(in_prose)
                    renamed = _variant(target, rng)
                    if renamed == target:
                        raise LitRPGInjectionError(f"no variant for {target!r}")
                    text, span = _replace_once(text, target, renamed, in_body=True)
                    # Within-chapter: the canonical name is in this chapter's own block.
                    defect = (target, renamed, False, chapter)
            except LitRPGInjectionError:
                available.append(chapter)  # unusable here; leave the chapter clean
                continue

            original, corrupt, cross, established = defect
            chapters[chapter] = text
            planted.append(
                LitRPGDefect(
                    defect_id=f"{manifest.manuscript_id}-{defect_type}-{chapter:03d}",
                    manuscript_id=manifest.manuscript_id,
                    type=defect_type,
                    chapter=chapter,
                    original_fragment=original,
                    corrupt_fragment=corrupt,
                    char_span=span,
                    clean_line=_line_containing(text, span),
                    cross_chapter=cross,
                    established_in=established,
                )
            )

    # Join once, then shift every span by its chapter's offset. Doing it here rather than
    # during injection means no span is ever stale.
    out: list[str] = []
    offsets: dict[int, int] = {}
    cursor = 0
    for state in manifest.chapters:
        offsets[state.chapter] = cursor
        body = chapters[state.chapter]
        out.append(body)
        cursor += len(body) + 1  # +1 for the newline the join inserts

    shifted = [
        LitRPGDefect(
            **{
                **d.as_dict(),
                "char_span": (
                    d.char_span[0] + offsets[d.chapter],
                    d.char_span[1] + offsets[d.chapter],
                ),
            }
        )
        for d in planted
    ]
    text = "\n".join(out)
    # Stamp the shortfall on every defect. A run that planted 4 of 10 returns the same
    # shape as one that planted 10, so recall quoted against the requested count would be
    # silently wrong — and a denominator that was never planted is the exact shape of the
    # survivorship bugs this project has already hit twice.
    note = f"planted {len(shifted)} of {requested} requested"
    shifted = [LitRPGDefect(**{**d.as_dict(), "notes": note}) for d in shifted]

    # Verify every span survived the join. Status-block fragments are stored stripped for
    # readability while the span covers the indent, so compare stripped — but compare, and
    # fail loudly. A stale offset would misalign scoring silently, and a recall number
    # computed against the wrong spans still looks like a recall number.
    for defect in shifted:
        found = text[defect.char_span[0] : defect.char_span[1]].strip()
        if found != defect.corrupt_fragment.strip():
            raise LitRPGInjectionError(
                f"{defect.defect_id}: span points at {found!r}, expected "
                f"{defect.corrupt_fragment!r}. Offsets are wrong; scoring would be silently "
                "misaligned rather than fail."
            )
    return text, shifted


def render_clean(manifest: Manifest) -> str:
    """The uncorrupted manuscript, byte-identical to what injection starts from."""
    return "\n".join(render_chapter(manifest, c.chapter) for c in manifest.chapters)


_ = render_status_block  # re-exported indirectly via litrpg; referenced for clarity
