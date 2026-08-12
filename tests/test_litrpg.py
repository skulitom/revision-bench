"""Tests for the synthetic LitRPG stratum.

The most important test here is :func:`test_detectors_do_not_import_the_injector`, for the
same reason as in `test_detect.py`: a detector that can see how a defect was planted is
measuring the injector, not the text, and every recall number it produces is fiction.

The second most important are the two that pin bugs found while building this, both of
which produced a *plausible number* rather than a failure:

- the injector replaced the first occurrence of a skill name, which is inside the status
  block rather than the prose. The planted defect then agreed with the canonical record and
  was undetectable by construction, so `skill_before_acquisition` scored 0% recall and
  looked like a detector problem.
- `inventory_ghost` chose items held in an adjacent chapter. Prose may legitimately name an
  item in the chapter it is lost, so those were not contradictions at all, and counting them
  as misses understated the detector.
"""

from __future__ import annotations

import ast
import pathlib
from itertools import pairwise

import pytest

from revisionbench.litrpg import STAT_NAMES, build_manifest, render_chapter, render_manuscript
from revisionbench.litrpg_detect import (
    LITRPG_DETECTORS,
    detect_all_litrpg,
    detect_level_regression,
    detect_stat_drift,
    parse_chapters,
)
from revisionbench.litrpg_inject import (
    LITRPG_DEFECT_TYPES,
    inject_manuscript,
)


def test_detectors_do_not_import_the_injector() -> None:
    """The detector must not be able to see how defects were planted."""
    tree = ast.parse(pathlib.Path("revisionbench/litrpg_detect.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("inject" in name for name in imported), imported


class TestManifest:
    def test_is_deterministic_in_id_and_seed(self) -> None:
        # Seeding from hash() of the id would regenerate a different world every process;
        # that bug already cost this project a corpus once (see corpus.py).
        a = build_manifest("ms-01", chapters=8, seed=3)
        b = build_manifest("ms-01", chapters=8, seed=3)
        assert a.as_dict() == b.as_dict()
        assert build_manifest("ms-02", chapters=8, seed=3).as_dict() != a.as_dict()

    def test_level_never_falls_and_rises_by_one(self) -> None:
        chapters = build_manifest("ms-01", chapters=20, seed=0).chapters
        for previous, current in pairwise(chapters):
            assert current.level - previous.level in (0, 1)

    def test_stats_change_only_on_level_up(self) -> None:
        """The invariant every stat_drift defect violates, so it defines 'clean'."""
        chapters = build_manifest("ms-01", chapters=20, seed=0).chapters
        for previous, current in pairwise(chapters):
            if current.level == previous.level:
                assert current.stats == previous.stats, current.chapter

    def test_skills_accumulate_and_are_never_lost(self) -> None:
        chapters = build_manifest("ms-01", chapters=20, seed=0).chapters
        for previous, current in pairwise(chapters):
            assert set(previous.skills) <= set(current.skills)

    def test_needs_at_least_two_chapters(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            build_manifest("ms-01", chapters=1)

    def test_rendered_status_block_reports_the_manifest(self) -> None:
        manifest = build_manifest("ms-01", chapters=6, seed=0)
        state = manifest.state_at(4)
        block = render_chapter(manifest, 4)
        assert f"Level: {state.level}" in block
        for stat in STAT_NAMES:
            assert f"{stat}: {state.stats[stat]}" in block


class TestCleanTextIsClean:
    """No detector may fire on an uncorrupted manuscript, at any seed.

    This is the precision floor. A detector that complains about clean text would make
    every recall number meaningless, because a harness cannot tell a real complaint from a
    reflex — and the first version of `detect_entity_rename` raised 20 per manuscript by
    matching any token of a canonical name against the following word.
    """

    @pytest.mark.parametrize("seed", range(6))
    def test_no_complaints_on_clean_manuscripts(self, seed: int) -> None:
        manifest = build_manifest(f"ms-{seed:02d}", chapters=14, seed=seed)
        assert detect_all_litrpg(render_manuscript(manifest)) == []


class TestInjection:
    def test_spans_survive_the_join(self) -> None:
        """Every defect's span must point at its own corruption in the final text."""
        manifest = build_manifest("ms-01", chapters=14, seed=0)
        text, defects = inject_manuscript(manifest, per_type=2, seed=0)
        assert defects
        for defect in defects:
            found = text[defect.char_span[0] : defect.char_span[1]]
            assert found.strip() == defect.corrupt_fragment.strip(), defect.defect_id

    def test_prose_defects_do_not_edit_the_status_block(self) -> None:
        """The bug that silently zeroed two defect types' recall.

        A skill's first occurrence in a chapter is in the ``Skills:`` line. Corrupting that
        changes the canonical record to agree with the corruption, which is not a
        contradiction at all.
        """
        manifest = build_manifest("ms-01", chapters=14, seed=0)
        text, defects = inject_manuscript(
            manifest, types=["skill_before_acquisition", "entity_rename"], per_type=2, seed=0
        )
        readings = {r.chapter: r for r in parse_chapters(text)}
        for defect in defects:
            reading = readings[defect.chapter]
            assert defect.char_span[0] >= reading.body_start, (
                f"{defect.defect_id} corrupted the status block, not the prose"
            )

    def test_unknown_defect_type_is_rejected(self) -> None:
        manifest = build_manifest("ms-01", chapters=6, seed=0)
        with pytest.raises(ValueError, match="unknown defect type"):
            inject_manuscript(manifest, types=["nonsense"])

    def test_one_defect_per_chapter(self) -> None:
        """Two contradictions in one chapter can interact, leaving an unscoreable target."""
        manifest = build_manifest("ms-01", chapters=14, seed=0)
        _, defects = inject_manuscript(manifest, per_type=2, seed=0)
        chapters = [d.chapter for d in defects]
        assert len(chapters) == len(set(chapters))

    def test_a_shortfall_is_recorded_rather_than_silent(self) -> None:
        """Planting fewer than requested is fine; hiding it is not.

        A run that planted 4 of 10 and one that planted 10 return the same shape, so a
        recall figure quoted against the requested count would silently be wrong. Every
        defect carries the shortfall.
        """
        manifest = build_manifest("ms-01", chapters=3, seed=0)
        _, defects = inject_manuscript(manifest, per_type=20, seed=0)
        assert len(defects) < 100
        assert all(f"of {5 * 20} requested" in d.notes for d in defects)


class TestDetectors:
    def test_level_regression_is_found_without_any_canonical_facts(self) -> None:
        """The claim that matters: the manuscript contradicts itself, no manifest needed."""
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        complaints = detect_level_regression(text)
        assert len(complaints) == 1
        assert "falls from 5" in complaints[0].message

    def test_level_skip_is_also_a_complaint(self) -> None:
        text = "Chapter 1\n  Level: 2\n\nprose.\n\nChapter 2\n  Level: 7\n\nprose.\n"
        assert "jumps" in detect_level_regression(text)[0].message

    def test_stat_change_without_a_level_up_is_a_complaint(self) -> None:
        text = (
            "Chapter 1\n  Level: 2\n  Strength: 10\n\nprose.\n\n"
            "Chapter 2\n  Level: 2\n  Strength: 14\n\nprose.\n"
        )
        assert len(detect_stat_drift(text)) == 1

    def test_stat_change_with_a_level_up_is_allowed(self) -> None:
        text = (
            "Chapter 1\n  Level: 2\n  Strength: 10\n\nprose.\n\n"
            "Chapter 2\n  Level: 3\n  Strength: 14\n\nprose.\n"
        )
        assert detect_stat_drift(text) == []

    def test_an_item_named_in_the_chapter_it_is_lost_is_not_a_ghost(self) -> None:
        """Prose that says a thing is gone necessarily names it."""
        text = (
            "Chapter 1\n  Level: 1\n  Inventory: iron key\n\nHe had the iron key.\n\n"
            "Chapter 2\n  Level: 1\n  Inventory: (empty)\n\nThe iron key was gone.\n"
        )
        assert [c for c in detect_all_litrpg(text) if c.type == "inventory_ghost"] == []

    def test_an_item_never_held_nearby_is_a_ghost(self) -> None:
        text = (
            "Chapter 1\n  Level: 1\n  Inventory: iron key\n\nHe had the iron key.\n\n"
            "Chapter 2\n  Level: 1\n  Inventory: iron key\n\nHe had the bone charm.\n\n"
            "Chapter 3\n  Level: 1\n  Inventory: iron key, bone charm\n\nprose.\n"
        )
        ghosts = [c for c in detect_all_litrpg(text) if c.type == "inventory_ghost"]
        assert len(ghosts) == 1

    def test_unknown_detector_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown detector"):
            detect_all_litrpg("Chapter 1\n  Level: 1\n", types=["nope"])

    def test_every_defect_type_has_a_detector(self) -> None:
        """A planted type with no detector scores 0% and looks like a hard problem."""
        assert set(LITRPG_DEFECT_TYPES) <= set(LITRPG_DETECTORS)
