"""Find contradictions in a LitRPG manuscript, from the manuscript alone.

The point of this module is what it does *not* need. `harness-gap.md` §2.1 claims the defect
classes that matter to a book are "mechanically detectable given a record of what is
canonical", and names a canonical-facts store as the prerequisite. In this genre the
manuscript carries its own record: the status blocks state the level, stats, skills and
inventory outright. So the canonical facts can be **extracted from the text** rather than
supplied, and the whole cross-chapter consistency problem reduces to parsing plus
bookkeeping — no judge, no model call, no author-maintained database.

That is the strongest form of the claim, so it is the mode that matters, and it is the
default. Passing a manifest enables ``oracle`` mode, which measures the same detectors given
perfect canonical facts. The gap between the two is the cost of having to infer the facts,
and reporting both is the honest way to present either.

Never imports the injector — enforced by test, as with `detect.py`. A detector that knows
how defects were planted measures the injector rather than the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

from revisionbench.detect import Complaint
from revisionbench.litrpg import STAT_NAMES
from revisionbench.text import normalise_newlines

__all__ = [
    "KNOWN_FIELDS",
    "LITRPG_DETECTORS",
    "ChapterReading",
    "detect_all_litrpg",
    "detect_entity_rename",
    "detect_inventory_ghost",
    "detect_level_regression",
    "detect_skill_before_acquisition",
    "detect_stat_drift",
    "parse_chapters",
]

_CHAPTER_RE = re.compile(r"^Chapter (\d+)\s*$", re.MULTILINE)

#: Indentation is not part of the schema. A model asked to reproduce a status block
#: reproduces the *fields* and quietly renormalises the whitespace — phi4 returns
#: ``Name: Bright`` where the template has ``  Name: Bright``. Requiring the two spaces made
#: the parser find no fields at all, which raises no complaints, which reads as perfect
#: precision. Real serials vary formatting far more than this, so tolerance here is
#: realism rather than leniency.
_FIELD_RE = re.compile(r"^[ \t>*|-]*([A-Za-z]+)\s*:[ \t]*(.+?)[ \t]*$", re.MULTILINE)

#: ...but tolerance needs a bound, or a line of dialogue like ``Bex: not today`` becomes a
#: canonical fact. The harness knows its own schema, so only these names are read.
KNOWN_FIELDS = frozenset({"Name", "Level", "Skills", "Inventory", *STAT_NAMES})


@dataclass(frozen=True, slots=True)
class ChapterReading:
    """What one chapter's status block asserts, plus where the chapter sits in the text."""

    chapter: int
    start: int
    end: int
    level: int | None
    stats: dict[str, int]
    skills: tuple[str, ...]
    inventory: tuple[str, ...]
    block_spans: dict[str, tuple[int, int]]
    body_start: int


def parse_chapters(text: str) -> list[ChapterReading]:
    """Read every chapter's status block. Tolerant: a missing field is None, not an error."""
    text = normalise_newlines(text)
    marks = [(int(m.group(1)), m.start()) for m in _CHAPTER_RE.finditer(text)]
    readings: list[ChapterReading] = []
    for i, (number, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        chunk = text[start:end]
        level: int | None = None
        stats: dict[str, int] = {}
        skills: tuple[str, ...] = ()
        inventory: tuple[str, ...] = ()
        spans: dict[str, tuple[int, int]] = {}
        body_start = end

        for match in _FIELD_RE.finditer(chunk):
            key, raw = match.group(1), match.group(2).strip()
            if key not in KNOWN_FIELDS:
                continue
            spans[key] = (start + match.start(), start + match.end())
            body_start = min(body_start, start + match.end())
            if key == "Level":
                level = int(raw) if raw.lstrip("-").isdigit() else None
            elif key == "Skills":
                skills = tuple(s.strip() for s in raw.split(",") if s.strip())
            elif key == "Inventory":
                inventory = (
                    ()
                    if raw == "(empty)"
                    else tuple(s.strip() for s in raw.split(",") if s.strip())
                )
            elif key == "Name":
                continue
            elif raw.lstrip("-").isdigit():
                stats[key] = int(raw)

        # Prose begins after the last status field, so a skill named inside the block is
        # not mistaken for a use of it in the narrative.
        last_field = max((s[1] for s in spans.values()), default=start)
        readings.append(
            ChapterReading(
                chapter=number,
                start=start,
                end=end,
                level=level,
                stats=stats,
                skills=skills,
                inventory=inventory,
                block_spans=spans,
                body_start=last_field,
            )
        )
    return readings


def detect_level_regression(text: str, **_: object) -> list[Complaint]:
    """Level must not fall, and must not jump by more than one.

    Needs no canonical facts at all — the sequence contradicts itself.
    """
    out: list[Complaint] = []
    readings = [r for r in parse_chapters(text) if r.level is not None]
    for previous, current in pairwise(readings):
        assert previous.level is not None and current.level is not None
        delta = current.level - previous.level
        if delta < 0:
            out.append(
                Complaint(
                    type="level_regression",
                    span=current.block_spans.get("Level", (current.start, current.start)),
                    message=(
                        f"level falls from {previous.level} (chapter {previous.chapter}) to "
                        f"{current.level} (chapter {current.chapter})"
                    ),
                    evidence=f"chapter {previous.chapter}: Level {previous.level}",
                )
            )
        elif delta > 1:
            out.append(
                Complaint(
                    type="level_regression",
                    span=current.block_spans.get("Level", (current.start, current.start)),
                    message=(
                        f"level jumps {previous.level} -> {current.level} between chapters "
                        f"{previous.chapter} and {current.chapter}"
                    ),
                    evidence=f"chapter {previous.chapter}: Level {previous.level}",
                )
            )
    return out


def detect_stat_drift(text: str, **_: object) -> list[Complaint]:
    """A stat may only change in a chapter where the level also rises.

    The rule is what makes this checkable without a manifest. A stat that moves on its own
    is either an error or an unstated in-world event, and a harness is entitled to ask.
    """
    out: list[Complaint] = []
    readings = parse_chapters(text)
    for previous, current in pairwise(readings):
        levelled = (
            previous.level is not None
            and current.level is not None
            and current.level > previous.level
        )
        if levelled:
            continue
        for stat, value in current.stats.items():
            if stat in previous.stats and previous.stats[stat] != value:
                out.append(
                    Complaint(
                        type="stat_drift",
                        span=current.block_spans.get(stat, (current.start, current.start)),
                        message=(
                            f"{stat} changes {previous.stats[stat]} -> {value} between "
                            f"chapters {previous.chapter} and {current.chapter} with no level-up"
                        ),
                        evidence=f"chapter {previous.chapter}: {stat} {previous.stats[stat]}",
                    )
                )
    return out


def _mentions(body: str, term: str, offset: int) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(re.escape(term), body):
        spans.append((offset + match.start(), offset + match.end()))
    return spans


def detect_skill_before_acquisition(text: str, **_: object) -> list[Complaint]:
    """A skill used in the prose before any status block lists it as known."""
    out: list[Complaint] = []
    readings = parse_chapters(text)
    known_by: dict[str, int] = {}
    for reading in readings:
        for skill in reading.skills:
            known_by.setdefault(skill, reading.chapter)

    for reading in readings:
        body = text[reading.body_start : reading.end]
        for skill, first_chapter in known_by.items():
            if first_chapter <= reading.chapter or skill in reading.skills:
                continue
            for span in _mentions(body, skill, reading.body_start):
                out.append(
                    Complaint(
                        type="skill_before_acquisition",
                        span=span,
                        message=(
                            f"{skill!r} is used in chapter {reading.chapter} but is first "
                            f"listed as known in chapter {first_chapter}"
                        ),
                        evidence=f"chapter {reading.chapter} skills: {', '.join(reading.skills)}",
                    )
                )
    return out


def detect_inventory_ghost(text: str, **_: object) -> list[Complaint]:
    """An item referenced in prose that the character does not hold.

    An item is referenceable in the chapter it is *lost* as well as while held — the prose
    that says it is gone necessarily names it. Without that allowance every loss would be a
    false positive, which is the difference between a usable detector and a noisy one.
    """
    out: list[Complaint] = []
    readings = parse_chapters(text)
    every_item = {item for r in readings for item in r.inventory}

    for index, reading in enumerate(readings):
        held = set(reading.inventory)
        if index > 0:
            held |= set(readings[index - 1].inventory)  # the chapter it went missing
        body = text[reading.body_start : reading.end]
        for item in every_item - held:
            for span in _mentions(body, item, reading.body_start):
                out.append(
                    Complaint(
                        type="inventory_ghost",
                        span=span,
                        message=(
                            f"{item!r} is referenced in chapter {reading.chapter} but is not "
                            "in inventory here or in the previous chapter"
                        ),
                        evidence=(
                            f"chapter {reading.chapter} inventory: "
                            f"{', '.join(reading.inventory) or '(empty)'}"
                        ),
                    )
                )
    return out


@lru_cache(maxsize=1)
def _function_words() -> frozenset[str]:
    """The repo's closed-class function-word list, cached.

    Loaded lazily so importing this module does not read a YAML file, and cached so the
    detector does not re-read it once per candidate.
    """
    from revisionbench.metrics.stylometry import load_function_words

    return frozenset(load_function_words()[1])


def _shape(name: str) -> tuple[bool, ...]:
    """Per-token capitalisation. ``Glass Song`` -> (True, True); ``cracked whetstone`` ->
    (False, False). A variant of a proper noun is still a proper noun."""
    return tuple(token[:1].isupper() for token in name.split())


def detect_entity_rename(text: str, **_: object) -> list[Complaint]:
    """A prose name that is one token away from a canonical skill or item.

    The status blocks are the authority on what things are called, so a near-variant in the
    narrative is drift. Requiring a shared token keeps this from firing on every unrelated
    noun phrase; requiring a *differing* token keeps it from firing on the name itself.
    """
    out: list[Complaint] = []
    readings = parse_chapters(text)
    canonical = {name for r in readings for name in (*r.skills, *r.inventory)}
    lowered = {n.lower() for n in canonical}

    # Anchor on the name's *leading* tokens and look at what follows. Matching on any token
    # instead fires on the canonical name itself: "Frost Nail without thinking" contains
    # "Nail without", which shares a token with "Frost Nail" and is not a variant of
    # anything. That produced 20 false positives per manuscript on clean text.
    prefixes: dict[str, set[str]] = {}
    for name in canonical:
        tokens = name.split()
        if len(tokens) < 2:
            continue
        prefixes.setdefault(" ".join(tokens[:-1]), set()).add(name)

    seen: set[tuple[int, int]] = set()
    for reading in readings:
        body = text[reading.body_start : reading.end]
        for prefix, names in prefixes.items():
            # Case-SENSITIVE, and the candidate must share the canonical name's
            # capitalisation shape. Both are load-bearing on real prose and neither
            # mattered on templated prose, which is why this passed at 88% precision there
            # and 23% here: "Glass Song" matched "glass shards" and "glass with", and
            # "Silent Palm" matched "silent enigma". 118 false positives across 8
            # manuscripts, all of them this. A named skill is a proper noun; a lowercase
            # match is ordinary prose using an ordinary word.
            #
            # Shape rather than a stop-word list on purpose: `detect.py` holds the line that
            # a detector needs no dictionary or cast list, and a rule that reads the text's
            # own capitalisation keeps that property.
            for match in re.finditer(rf"\b{re.escape(prefix)} ([A-Za-z]+)", body):
                candidate = match.group(0)
                if candidate.lower() in lowered:
                    continue  # the canonical name itself
                if not all(
                    token[:1].isupper() == expected
                    for token, expected in zip(
                        candidate.split(), _shape(sorted(names)[0]), strict=False
                    )
                ):
                    continue
                # Capitalisation saves the Title Case skills and does nothing for lowercase
                # item names: "salt pouch" still fired on "salt from", "salt on", "salt and".
                # A closed-class function word can never be the second half of an item name,
                # so a candidate ending in one is prose. The list is the same versioned
                # closed class Burrows' Delta uses — a fixed grammatical category, not the
                # open-ended dictionary `detect.py` refuses to depend on.
                if candidate.split()[-1].lower() in _function_words():
                    continue
                span = (reading.body_start + match.start(), reading.body_start + match.end())
                if span in seen:
                    continue
                seen.add(span)
                out.append(
                    Complaint(
                        type="entity_rename",
                        span=span,
                        message=f"{candidate!r} looks like a variant of {sorted(names)[0]!r}",
                        evidence=f"canonical: {', '.join(sorted(names))}",
                    )
                )
    return out


LITRPG_DETECTORS = {
    "level_regression": detect_level_regression,
    "stat_drift": detect_stat_drift,
    "skill_before_acquisition": detect_skill_before_acquisition,
    "inventory_ghost": detect_inventory_ghost,
    "entity_rename": detect_entity_rename,
}


def detect_all_litrpg(text: str, *, types: list[str] | None = None) -> list[Complaint]:
    """Run every LitRPG detector, ordered by position.

    Raises:
        ValueError: An unknown detector name — a typo must not silently reduce coverage.
    """
    chosen = list(types or LITRPG_DETECTORS)
    unknown = [t for t in chosen if t not in LITRPG_DETECTORS]
    if unknown:
        raise ValueError(f"unknown detector(s): {unknown}; known: {sorted(LITRPG_DETECTORS)}")
    normalised = normalise_newlines(text)
    out: list[Complaint] = []
    for name in chosen:
        out.extend(LITRPG_DETECTORS[name](normalised))
    return sorted(out, key=lambda c: c.span)
