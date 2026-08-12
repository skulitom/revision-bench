"""A synthetic LitRPG manuscript with exact, machine-checkable ground truth.

Every stratum in this project so far starts from real prose and annotates it. This one
inverts that, and the inversion is the point.

**Why a genre corpus at all.** `harness-gap.md` §4 names the gap that matters for a book:
the defects that hurt a manuscript most are *non-local* — a name established in chapter 1
drifting in chapter 12, a number that only contradicts itself two hundred pages later. Every
metric in this repo is within-passage, because the corpus is ten isolated 900-word passages.
Nothing here can currently even express a cross-chapter contradiction, let alone detect one.

**Why LitRPG specifically.** The genre carries explicit, enumerable state — levels, stats,
skill names, inventory — so "is this manuscript self-consistent" becomes a diff against a
table rather than a matter of opinion. §2.1 of the same document argues most defect classes
that matter need no judge; LitRPG is the genre where that argument is most obviously true.
Two of the five Stratum B defect types, ``name_drift`` and ``number_drift``, are the genre's
actual failure modes.

**Why synthetic, and what that costs.** There is no licensed LitRPG corpus: the genre is
roughly 2015-onward, so nothing is near public domain, and web-serial chapters belong to
their authors. Rather than build on text we cannot redistribute, the world is generated
first — as a state machine — and the prose is rendered *from* it. The manifest is therefore
not an annotation of the text but its source, which makes the ground truth exact instead of
estimated.

What that buys: cross-chapter contradictions with known spans, at any scale, no licensing
question, and a clean separation between the facts and the prose that states them.

What it costs, stated plainly so no result built on this overclaims:

- **It says nothing about prose quality.** The clean text is not good writing and is not
  claimed to be. This stratum measures whether contradictions are *findable*, not whether
  revisions are *better*. Any quality claim needs Stratum A.
- **The prose is regular in ways real prose is not**, so detector precision measured here is
  an upper bound. A detector tuned until it is perfect on this will not be perfect on a real
  manuscript.

`render_chapter` produces templated prose; `prompt_for_chapter` produces the instruction for
a model to write the same chapter instead. The manifest is ground truth either way, which is
the property that makes the swap safe.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "MANIFEST_VERSION",
    "STAT_NAMES",
    "ChapterState",
    "Manifest",
    "WorldFact",
    "build_manifest",
    "prompt_for_chapter",
    "render_chapter",
    "render_manuscript",
]

#: Bumped whenever a change here could alter a generated world or its rendering. Stamped on
#: the manifest, so a corpus built by two versions is detectable rather than silently mixed.
MANIFEST_VERSION = 1

STAT_NAMES = ("Strength", "Agility", "Intellect", "Vitality")

_PROTAGONISTS = ("Kaelen", "Sorrel", "Bright", "Vance", "Ilrid", "Tam", "Odile", "Rhys")
_SKILLS = (
    "Ember Lash",
    "Stone Ward",
    "Quickstep",
    "Mind Spike",
    "Iron Skin",
    "Frost Nail",
    "Echo Sight",
    "Blood Ledger",
    "Silent Palm",
    "Thorn Coil",
    "Ash Beckon",
    "Glass Song",
)
_ITEMS = (
    "cracked whetstone",
    "brass compass",
    "sealed vial",
    "iron key",
    "bone charm",
    "salt pouch",
    "copper ring",
    "folded map",
    "tin lantern",
    "grey cloak",
)
_LOCATIONS = (
    "the Drowned Steps",
    "Kellsmarket",
    "the Ninth Terrace",
    "Underbridge",
    "the Sallow Wood",
    "Fenmouth",
    "the Cistern Gate",
    "Highcut",
)
_NPCS = ("Maro", "Sister Elspeth", "the Tallyman", "Bex", "Warden Ott", "Nel")


@dataclass(frozen=True, slots=True)
class WorldFact:
    """One canonical fact, with the chapter it becomes true.

    Facts are the unit a detector is allowed to know about. Keeping them separate from the
    rendering is what lets `litrpg_detect` consume ground truth without ever seeing how the
    prose — or a corruption of it — was produced.
    """

    kind: str  # level | stat | skill | item_gained | item_lost | location | npc
    key: str
    value: str
    chapter: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChapterState:
    """The protagonist's complete state at the *end* of a chapter."""

    chapter: int
    level: int
    stats: dict[str, int]
    skills: tuple[str, ...]
    inventory: tuple[str, ...]
    location: str
    npc: str
    #: Set when this chapter contains a level-up, because a stat may only change here. A
    #: stat that moves in a chapter without one is a contradiction by construction, and
    #: that rule is what makes `stat_drift` detectable rather than merely suspicious.
    levelled_up: bool = False
    skill_gained: str = ""
    item_gained: str = ""
    item_lost: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["skills"] = list(self.skills)
        data["inventory"] = list(self.inventory)
        return data


@dataclass(frozen=True, slots=True)
class Manifest:
    """A whole manuscript's ground truth."""

    manuscript_id: str
    protagonist: str
    chapters: tuple[ChapterState, ...]
    facts: tuple[WorldFact, ...] = field(default=())
    seed: int = 0
    manifest_version: int = MANIFEST_VERSION

    def state_at(self, chapter: int) -> ChapterState:
        return self.chapters[chapter - 1]

    def skills_known_at(self, chapter: int) -> frozenset[str]:
        """Skills legitimately usable in this chapter — the causality check's ground truth."""
        return frozenset(self.state_at(chapter).skills)

    def inventory_at(self, chapter: int) -> frozenset[str]:
        return frozenset(self.state_at(chapter).inventory)

    def entities(self) -> frozenset[str]:
        """Every proper name the manuscript may use. A name outside this set is drift."""
        names = {self.protagonist}
        for state in self.chapters:
            names.update(state.skills)
            names.update(state.inventory)
            names.add(state.location)
            names.add(state.npc)
        return frozenset(names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manuscript_id": self.manuscript_id,
            "protagonist": self.protagonist,
            "seed": self.seed,
            "manifest_version": self.manifest_version,
            "chapters": [c.as_dict() for c in self.chapters],
            "facts": [f.as_dict() for f in self.facts],
        }


def _rng(seed: int, manuscript_id: str) -> random.Random:
    """Seed from a hash of the id, not ``hash()``.

    Python randomises string hashing per process, so seeding from ``hash(manuscript_id)``
    would generate a different world on every run. That exact bug already cost this project
    a corpus once — see `corpus.py`.
    """
    digest = hashlib.sha256(f"{seed}:{manuscript_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_manifest(
    manuscript_id: str, *, chapters: int = 12, seed: int = 0, level_every: int = 3
) -> Manifest:
    """Generate a self-consistent world. Deterministic in ``(manuscript_id, seed)``.

    The invariants below are what every planted defect violates, so they are the definition
    of "clean" for this stratum:

    - level rises by exactly one, only on a level-up chapter, and never falls
    - a stat changes only in a level-up chapter
    - skills accumulate and are never silently lost
    - an item is in inventory from the chapter it is gained until the chapter it is lost
    """
    if chapters < 2:
        raise ValueError("a cross-chapter stratum needs at least 2 chapters")
    rng = _rng(seed, manuscript_id)

    protagonist = rng.choice(_PROTAGONISTS)
    skill_pool = list(_SKILLS)
    item_pool = list(_ITEMS)
    rng.shuffle(skill_pool)
    rng.shuffle(item_pool)

    level = 1
    stats = {name: rng.randint(8, 13) for name in STAT_NAMES}
    skills: list[str] = [skill_pool.pop()]
    inventory: list[str] = [item_pool.pop()]

    states: list[ChapterState] = []
    facts: list[WorldFact] = [
        WorldFact("level", "level", str(level), 1),
        WorldFact("skill", skills[0], "known", 1),
        WorldFact("item_gained", inventory[0], "held", 1),
        *(WorldFact("stat", k, str(v), 1) for k, v in stats.items()),
    ]

    for chapter in range(1, chapters + 1):
        levelled = chapter > 1 and chapter % level_every == 1
        skill_gained = item_gained = item_lost = ""

        if levelled:
            level += 1
            bumped = rng.choice(STAT_NAMES)
            stats = {**stats, bumped: stats[bumped] + rng.randint(1, 2)}
            facts.append(WorldFact("level", "level", str(level), chapter))
            facts.append(WorldFact("stat", bumped, str(stats[bumped]), chapter))
            if skill_pool:
                skill_gained = skill_pool.pop()
                skills = [*skills, skill_gained]
                facts.append(WorldFact("skill", skill_gained, "known", chapter))

        # Item churn on non-level chapters, so gains and losses are not confounded with
        # level-ups — a detector could otherwise "find" inventory errors by tracking level.
        if not levelled and chapter > 1:
            if len(inventory) > 1 and rng.random() < 0.35:
                item_lost = rng.choice(inventory)
                inventory = [i for i in inventory if i != item_lost]
                facts.append(WorldFact("item_lost", item_lost, "gone", chapter))
            elif item_pool and rng.random() < 0.55:
                item_gained = item_pool.pop()
                inventory = [*inventory, item_gained]
                facts.append(WorldFact("item_gained", item_gained, "held", chapter))

        states.append(
            ChapterState(
                chapter=chapter,
                level=level,
                stats=dict(stats),
                skills=tuple(skills),
                inventory=tuple(inventory),
                location=rng.choice(_LOCATIONS),
                npc=rng.choice(_NPCS),
                levelled_up=levelled,
                skill_gained=skill_gained,
                item_gained=item_gained,
                item_lost=item_lost,
            )
        )

    return Manifest(
        manuscript_id=manuscript_id,
        protagonist=protagonist,
        chapters=tuple(states),
        facts=tuple(facts),
        seed=seed,
    )


def render_status_block(state: ChapterState, protagonist: str) -> str:
    """The genre's status readout. Fixed schema, because a detector parses it.

    Real serials vary this constantly. Holding it fixed makes stat extraction near-perfect
    here and therefore makes measured detector precision an upper bound — which is stated
    in the module docstring and must be repeated wherever a number from this stratum is
    reported.
    """
    lines = [
        "[ STATUS ]",
        f"  Name: {protagonist}",
        f"  Level: {state.level}",
        *(f"  {name}: {state.stats[name]}" for name in STAT_NAMES),
        f"  Skills: {', '.join(state.skills)}",
        f"  Inventory: {', '.join(state.inventory) if state.inventory else '(empty)'}",
    ]
    return "\n".join(lines)


def render_chapter(manifest: Manifest, chapter: int) -> str:
    """Templated prose for one chapter, consistent with the manifest by construction."""
    state = manifest.state_at(chapter)
    who = manifest.protagonist
    body = [f"Chapter {chapter}", "", render_status_block(state, who), ""]

    body.append(
        f"{who} came into {state.location} with the light already going. "
        f"{state.npc} was waiting at the far end, and did not look pleased to be kept."
    )
    if state.levelled_up:
        body.append(
            f"The threshold broke somewhere behind {who}'s eyes, and the world resolved "
            f"one notch sharper. Level {state.level}. "
            + (
                f"{state.skill_gained} settled into place alongside the rest."
                if state.skill_gained
                else ""
            )
        )
    if state.item_gained:
        body.append(
            f"{state.npc} pressed a {state.item_gained} into {who}'s hand and said nothing."
        )
    if state.item_lost:
        body.append(f"The {state.item_lost} was gone by the time {who} thought to check for it.")

    # Use a known skill, so a use-before-acquisition defect has something to corrupt.
    body.append(
        f"{who} brought up {state.skills[-1]} without thinking about it, and the corridor "
        f"answered. Whatever {state.npc} had come to say went unsaid a while longer."
    )
    if state.inventory:
        body.append(
            f"Afterwards {who} counted what was left: the {state.inventory[0]}, and not much else."
        )
    return "\n".join(body) + "\n"


def render_manuscript(manifest: Manifest) -> str:
    return "\n".join(render_chapter(manifest, c.chapter) for c in manifest.chapters)


def prompt_for_chapter(manifest: Manifest, chapter: int) -> str:
    """Instruction for a model to write this chapter instead of templating it.

    The manifest stays ground truth, so a model-written manuscript is scored identically.
    The prompt states the facts as constraints rather than suggestions, and forbids
    inventing state, because a model that promotes its own invention to canon produces
    contradictions the manifest cannot adjudicate — those would score as detector false
    positives when they are really corpus errors.
    """
    state = manifest.state_at(chapter)
    return (
        "Write one chapter of a LitRPG web serial, 200-300 words, present-day casual "
        "register. Begin with this status block reproduced EXACTLY, then the prose:\n\n"
        f"{render_status_block(state, manifest.protagonist)}\n\n"
        f"Facts you must respect and must not contradict:\n"
        f"- The protagonist is {manifest.protagonist}, currently level {state.level}.\n"
        f"- Skills known: {', '.join(state.skills)}. Use ONLY these; inventing a skill "
        f"or using one not listed is an error.\n"
        f"- Items held: {', '.join(state.inventory) or 'nothing'}. Do not reference items "
        f"that are not listed.\n"
        f"- Location: {state.location}. Present character: {state.npc}.\n"
        + (
            f"- {manifest.protagonist} reaches level {state.level} in this chapter.\n"
            if state.levelled_up
            else ""
        )
        + (f"- {state.skill_gained} is learned in this chapter.\n" if state.skill_gained else "")
        + (f"- A {state.item_gained} is acquired in this chapter.\n" if state.item_gained else "")
        + (f"- The {state.item_lost} is lost in this chapter.\n" if state.item_lost else "")
        + "\nDo not invent numeric stats, levels, skills or items beyond those listed. "
        "Output the chapter only."
    )
