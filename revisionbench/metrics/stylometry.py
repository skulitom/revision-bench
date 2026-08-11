"""M1 (stylometric identity) and M2 (homogenization) — plan.md §7.

The unit of measurement is a **style fingerprint**: a named vector of rate-valued
features, standardised against a fixed reference corpus, compared by mean absolute
z-difference. Burrows' Delta is the special case of that computed over function words
only, and it is exposed under its own name because that is the number the literature
recognises.

Three design decisions carry most of the risk.

**The scaler is fitted once and reused.** :meth:`StyleModel.fit` takes a reference corpus
— in Phase 0, the round-0 passages — and freezes the per-feature means and standard
deviations. Every later round is transformed with that frozen scaler and never re-fitted.
Re-fitting per round is the obvious convenience and it destroys the measurement: if the
whole corpus drifts toward one house style, re-standardising against the drifted corpus
subtracts exactly the effect H1 predicts, and homogenization comes out flat. The model
serialises (:meth:`to_dict`) so results can record which scaler produced them.

**Features are selected by reference-corpus frequency, from a closed class.** Rare
function words ("lest", "thine") have near-zero variance across a small reference corpus,
and dividing by that variance turns one incidental occurrence into a z-score of 12 that
dominates every Delta it appears in. Selecting the top-N by corpus frequency is Burrows'
own prescription; restricting the pool to closed-class words is this project's addition,
for the reason set out in ``data/function_words.yaml``.

**Degenerate text raises rather than scores.** A reviser that returns two sentences, or an
apology, or an empty string, must show up as a recorded failure, not as a passage with an
interesting fingerprint. :class:`MetricError` is the mechanism; the runner is expected to
catch it per round and record it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from revisionbench.metrics.stats import mean, median, percentile, stdev
from revisionbench.text import (
    PUNCTUATION_CLASS_NAMES,
    punctuation_counts,
    sentence_lengths,
    words,
)

__all__ = [
    "DEFAULT_FUNCTION_WORDS_PATH",
    "FEATURE_FAMILIES",
    "MIN_SENTENCES",
    "MIN_WORDS_FOR_MTLD",
    "MTLD_THRESHOLD",
    "MetricError",
    "StyleModel",
    "burrows_delta",
    "cluster_purity",
    "describe",
    "feature_vector",
    "load_function_words",
    "mean_pairwise_delta",
    "mtld",
    "punctuation_profile",
    "sentence_length_stats",
    "wasserstein1",
]


class MetricError(RuntimeError):
    """A text cannot be scored — too short, too degenerate, or structurally unusable."""


DEFAULT_FUNCTION_WORDS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "function_words.yaml"
)

#: McCarthy & Jarvis's canonical MTLD factor threshold. Not a tunable: changing it makes
#: values incomparable with every published MTLD, and with this repo's own earlier runs.
MTLD_THRESHOLD = 0.72

#: MTLD is unstable on short texts — with few tokens the partial factor dominates and the
#: value swings wildly. plan.md §5 passages are 500-1500 words, so this floor only fires
#: on genuinely degenerate revision output, which is exactly when it should.
MIN_WORDS_FOR_MTLD = 100

#: Below this, sentence-length spread is not estimable.
MIN_SENTENCES = 2

#: Feature-name prefixes, in the order a family-balanced distance weights them.
FEATURE_FAMILIES = ("fw", "punct", "sent", "lex")


# --------------------------------------------------------------------------------------
# Component metrics
# --------------------------------------------------------------------------------------


def load_function_words(path: str | Path | None = None) -> tuple[int, tuple[str, ...]]:
    """Load the closed-class function-word list. Returns ``(version, words)``.

    Raises:
        MetricError: The file is missing, malformed, or contains duplicates. Duplicates
            are fatal rather than deduplicated: a repeated word would be counted twice in
            the frequency ranking, and silently.
    """
    import yaml

    target = Path(path) if path is not None else DEFAULT_FUNCTION_WORDS_PATH
    if not target.is_file():
        raise MetricError(f"function-word list not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "words" not in data or "version" not in data:
        raise MetricError(f"{target}: expected a mapping with 'version' and 'words' keys")

    # The YAML 1.1 boolean trap, checked rather than trusted. PyYAML resolves bare `no`,
    # `yes`, `on`, `off`, `y`, `n`, `true` and `false` to booleans. Four of those -- no,
    # yes, on, off -- are function words this list needs, and `str(value).lower()` would
    # have turned them into the tokens "false" and "true", which occur in no passage. The
    # result is a Delta computed over a silently smaller feature set: no exception, no
    # warning, just a number that is quietly measuring less than it claims.
    bad = [w for w in data["words"] if not isinstance(w, str)]
    if bad:
        raise MetricError(
            f"{target}: {len(bad)} entr(y/ies) are not strings: {bad!r}. This is almost "
            f"certainly the YAML 1.1 boolean trap -- bare no/yes/on/off/y/n/true/false "
            f"parse as booleans. Quote them."
        )

    listed = [w.lower() for w in data["words"]]
    duplicates = sorted({w for w in listed if listed.count(w) > 1})
    if duplicates:
        raise MetricError(f"{target}: duplicate function words: {', '.join(duplicates)}")
    return int(data["version"]), tuple(listed)


def sentence_length_stats(text: str) -> dict[str, float]:
    """Sentence-length distribution summary (plan.md §7 M1).

    ``sent:count_per_1000w`` is included alongside the location and spread statistics
    because it is the one that moves when a reviser splits long sentences into short ones
    without changing the word count — the single most common "improvement" an LLM makes
    to literary prose, and the one that most directly erases a Woolf-shaped voice.

    Raises:
        MetricError: Fewer than :data:`MIN_SENTENCES` sentences.
    """
    lengths = [float(n) for n in sentence_lengths(text)]
    if len(lengths) < MIN_SENTENCES:
        raise MetricError(
            f"need at least {MIN_SENTENCES} sentences to describe a sentence-length "
            f"distribution, got {len(lengths)}; record this round as a metric failure "
            f"rather than scoring it"
        )
    total_words = sum(lengths)
    ordered = sorted(lengths)
    return {
        "sent:mean": mean(lengths),
        "sent:median": median(lengths),
        "sent:sd": stdev(lengths),
        "sent:p10": percentile(ordered, 0.10),
        "sent:p90": percentile(ordered, 0.90),
        # Guard against a text made entirely of empty "sentences" (a row of dashes).
        "sent:count_per_1000w": (len(lengths) / total_words * 1000.0) if total_words else 0.0,
    }


def punctuation_profile(text: str) -> dict[str, float]:
    """Punctuation rate per 1000 words, one entry per class (plan.md §7 M1).

    Rates rather than counts, because a reviser changes the length of the text it is
    revising and a raw comma count would then move for two different reasons at once.
    """
    counts = punctuation_counts(text)
    total = len(words(text))
    if total == 0:
        raise MetricError("cannot profile punctuation of a text with no words")
    scale = 1000.0 / total
    return {f"punct:{name}": counts[name] * scale for name in PUNCTUATION_CLASS_NAMES}


def mtld(text: str, *, threshold: float = MTLD_THRESHOLD) -> float:
    """Measure of Textual Lexical Diversity, bidirectional (plan.md §7 M1).

    Walks the token stream accumulating types until the running type-token ratio falls to
    ``threshold``, counts that as one "factor", and resets. MTLD is tokens per factor,
    averaged over a forward and a backward pass. Unlike raw TTR it is not a function of
    text length, which is what makes it usable across rounds that change the word count.

    Raises:
        MetricError: The text has fewer than :data:`MIN_WORDS_FOR_MTLD` tokens, or never
            completes a factor and ends at a type-token ratio of exactly 1.0 (every word
            distinct). The second case has no defined value: the naive formula divides by
            zero, and returning ``inf`` — or worse, the token count — would put an
            enormous "maximally diverse" number into a mean.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"MTLD threshold must be strictly between 0 and 1, got {threshold}")
    tokens = words(text)
    if len(tokens) < MIN_WORDS_FOR_MTLD:
        raise MetricError(
            f"MTLD needs at least {MIN_WORDS_FOR_MTLD} tokens to be stable, got "
            f"{len(tokens)}; record this round as a metric failure rather than scoring it"
        )
    forward = _mtld_pass(tokens, threshold)
    backward = _mtld_pass(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0


def _mtld_pass(tokens: Sequence[str], threshold: float) -> float:
    factors = 0.0
    types: set[str] = set()
    count = 0
    for token in tokens:
        count += 1
        types.add(token)
        if len(types) / count <= threshold:
            factors += 1.0
            types.clear()
            count = 0
    if count > 0:
        remaining_ttr = len(types) / count
        # A partial factor scales how far the tail got toward the threshold. At TTR 1.0
        # it contributes nothing, which is the degenerate case guarded below.
        factors += (1.0 - remaining_ttr) / (1.0 - threshold)
    if factors <= 0.0:
        raise MetricError(
            "MTLD is undefined for this text: no factor completed and every token is "
            "distinct, so tokens-per-factor divides by zero. This is degenerate input "
            "(a word list, not prose), not a diversity of infinity."
        )
    return len(tokens) / factors


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    """1-Wasserstein (earth mover's) distance between two 1-D empirical distributions.

    Used to compare sentence-length *distributions* across rounds rather than only their
    means. A reviser that replaces a mix of 5-word and 60-word sentences with a uniform
    diet of 25-word sentences can leave the mean untouched while flattening the rhythm
    completely; the mean cannot see that and this can.

    In the units of the inputs (words), so a value of 4.0 means "on average, sentence
    lengths must move 4 words to turn one distribution into the other".

    Raises:
        ValueError: Either sequence is empty.
    """
    if not a or not b:
        raise ValueError("wasserstein1 needs at least one value in each sample")
    xs = sorted(float(v) for v in a)
    ys = sorted(float(v) for v in b)
    grid = sorted(set(xs) | set(ys))
    if len(grid) == 1:
        return 0.0
    total = 0.0
    for left, right in itertools.pairwise(grid):
        # Fraction of each sample at or below `left`, times the width of the step.
        cdf_x = _fraction_at_or_below(xs, left)
        cdf_y = _fraction_at_or_below(ys, left)
        total += abs(cdf_x - cdf_y) * (right - left)
    return total


def _fraction_at_or_below(sorted_values: Sequence[float], threshold: float) -> float:
    import bisect

    return bisect.bisect_right(sorted_values, threshold) / len(sorted_values)


# --------------------------------------------------------------------------------------
# Feature vector
# --------------------------------------------------------------------------------------


def feature_vector(text: str, function_words: Iterable[str]) -> dict[str, float]:
    """Build the full named feature vector for one text.

    Families, by name prefix:

    - ``fw:<word>`` — function-word frequency per 1000 word tokens.
    - ``punct:<class>`` — punctuation rate per 1000 word tokens.
    - ``sent:<stat>`` — sentence-length location, spread and density.
    - ``lex:mtld`` — lexical diversity.

    Every requested function word gets an entry, including the ones that do not occur, so
    two vectors are always comparable key-for-key. An absent word is a measured zero here;
    a *missing key* would let a later dict comparison silently skip the feature that
    changed most.

    Raises:
        MetricError: The text is empty or too degenerate to score (see :func:`mtld` and
            :func:`sentence_length_stats`).
    """
    tokens = words(text)
    if not tokens:
        raise MetricError("cannot build a feature vector for a text with no word tokens")
    total = len(tokens)
    scale = 1000.0 / total

    tally: dict[str, int] = {}
    for token in tokens:
        tally[token] = tally.get(token, 0) + 1

    vector: dict[str, float] = {f"fw:{w}": tally.get(w, 0) * scale for w in function_words}
    vector.update(punctuation_profile(text))
    vector.update(sentence_length_stats(text))
    vector["lex:mtld"] = mtld(text)
    return vector


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StyleModel:
    """A frozen standardisation of the feature space, fitted to a reference corpus.

    Attributes:
        features: Feature names in a fixed order — the ones that survived selection and
            the variance floor.
        means: Reference-corpus mean per feature.
        sds: Reference-corpus standard deviation per feature.
        function_words: The function words whose ``fw:`` features were kept.
        function_words_version: Version of ``data/function_words.yaml`` used.
        n_reference_texts: How many texts the scaler was fitted on. Recorded because a
            scaler fitted on 10 texts carries real sampling noise, and a reader comparing
            two runs needs to know which one had the better-estimated scale.
        dropped_low_variance: Features excluded for having a reference SD at or below
            ``sd_floor``. Kept as a record rather than dropped silently: if this list is
            long, the reference corpus is too small or too homogeneous for the feature
            set, and Delta is being computed over fewer dimensions than it looks.
        sd_floor: The variance floor used.
    """

    features: tuple[str, ...]
    means: dict[str, float]
    sds: dict[str, float]
    function_words: tuple[str, ...]
    function_words_version: int
    n_reference_texts: int
    dropped_low_variance: tuple[str, ...]
    sd_floor: float

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        function_words: Sequence[str] | None = None,
        function_words_version: int | None = None,
        n_function_words: int = 100,
        sd_floor: float = 1e-9,
    ) -> StyleModel:
        """Fit the scaler to a reference corpus.

        Args:
            texts: Reference texts. In Phase 0 these are the round-0 passages. Two or more
                are required — a standard deviation needs a sample.
            function_words: Candidate pool. Defaults to ``data/function_words.yaml``.
            function_words_version: Version tag for the pool, when supplying it directly.
            n_function_words: Keep the top N by total reference-corpus frequency. Burrows'
                classic range is 100-150. Lower it if the reference corpus is small.
            sd_floor: Features whose reference SD is at or below this are dropped rather
                than divided by. A near-zero denominator turns a single incidental
                occurrence into a z-score in the double digits, and because Delta is a
                *mean* of absolute z-differences, one such feature can move the whole
                statistic on its own.

        Raises:
            MetricError: Fewer than two reference texts, or every feature was dropped.
        """
        if len(texts) < 2:
            raise MetricError(
                f"StyleModel.fit needs at least 2 reference texts to estimate a standard "
                f"deviation, got {len(texts)}"
            )
        if function_words is None:
            function_words_version, pool = load_function_words()
        else:
            pool = tuple(function_words)
            if function_words_version is None:
                raise MetricError(
                    "supply function_words_version alongside an explicit function_words "
                    "list; an unversioned feature set cannot be reproduced"
                )
        if n_function_words < 1:
            raise ValueError(f"n_function_words must be >= 1, got {n_function_words}")

        # Select by *total reference-corpus* frequency, not per-text, so the feature set
        # does not depend on the order texts happen to be passed in.
        pool_set = set(pool)
        corpus_tally: dict[str, int] = {}
        total_tokens = 0
        for text in texts:
            for token in words(text):
                total_tokens += 1
                if token in pool_set:
                    corpus_tally[token] = corpus_tally.get(token, 0) + 1
        if total_tokens == 0:
            raise MetricError("reference corpus contains no word tokens")

        # Ties broken alphabetically so the selection is deterministic.
        ranked = sorted(pool, key=lambda w: (-corpus_tally.get(w, 0), w))
        selected = tuple(w for w in ranked[:n_function_words] if corpus_tally.get(w, 0) > 0)
        if not selected:
            raise MetricError("no function word from the pool occurs in the reference corpus")

        vectors = [feature_vector(text, selected) for text in texts]
        names = sorted(vectors[0])
        means: dict[str, float] = {}
        sds: dict[str, float] = {}
        dropped: list[str] = []
        for name in names:
            column = [v[name] for v in vectors]
            sd = stdev(column)
            if sd <= sd_floor:
                dropped.append(name)
                continue
            means[name] = mean(column)
            sds[name] = sd
        if not means:
            raise MetricError(
                "every feature had zero variance across the reference corpus; the texts "
                "are identical or the corpus is too small to standardise against"
            )
        return cls(
            features=tuple(sorted(means)),
            means=means,
            sds=sds,
            function_words=selected,
            function_words_version=int(function_words_version or 0),
            n_reference_texts=len(texts),
            dropped_low_variance=tuple(dropped),
            sd_floor=sd_floor,
        )

    def zscores(self, text: str) -> dict[str, float]:
        """Standardise one text into the fitted feature space."""
        raw = feature_vector(text, self.function_words)
        return {name: (raw[name] - self.means[name]) / self.sds[name] for name in self.features}

    def delta(self, a: str, b: str, *, family: str | None = None) -> float:
        """Mean absolute z-difference between two texts.

        With ``family=None`` this is the plain mean over every retained feature, which
        makes it Burrows' Delta generalised to the full writeprint. Because the function
        word family has ~100 members and the others have ~16, ~6 and 1, that mean is
        dominated by function words; :meth:`family_balanced_delta` is the alternative.

        Args:
            family: Restrict to one family prefix (see :data:`FEATURE_FAMILIES`).
        """
        za, zb = self.zscores(a), self.zscores(b)
        names = self._family_features(family)
        return mean([abs(za[n] - zb[n]) for n in names])

    def family_balanced_delta(self, a: str, b: str) -> float:
        """Mean over families of the within-family mean absolute z-difference.

        Each of function words, punctuation, sentence shape and lexical diversity gets
        equal weight. This is the form to prefer for the plan.md §8 A4 voice veto: a
        reviser that leaves function words alone while replacing every dash with a comma
        and halving the sentence length has changed the voice, and a function-word-weighted
        mean would barely register it.
        """
        za, zb = self.zscores(a), self.zscores(b)
        per_family = []
        for family in FEATURE_FAMILIES:
            names = [n for n in self.features if n.startswith(f"{family}:")]
            if names:
                per_family.append(mean([abs(za[n] - zb[n]) for n in names]))
        if not per_family:
            raise MetricError("no families retained any features")
        return mean(per_family)

    def _family_features(self, family: str | None) -> list[str]:
        if family is None:
            return list(self.features)
        if family not in FEATURE_FAMILIES:
            raise ValueError(f"unknown feature family {family!r}; known: {FEATURE_FAMILIES}")
        names = [n for n in self.features if n.startswith(f"{family}:")]
        if not names:
            raise MetricError(
                f"family {family!r} retained no features after fitting; it cannot be "
                f"compared, and treating that as a distance of 0 would read as "
                f"'identical' rather than 'not measured'"
            )
        return names

    def to_dict(self) -> dict[str, Any]:
        """Serialise, so a result file records the exact scaler that produced its numbers."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StyleModel:
        return cls(
            features=tuple(data["features"]),
            means=dict(data["means"]),
            sds=dict(data["sds"]),
            function_words=tuple(data["function_words"]),
            function_words_version=int(data["function_words_version"]),
            n_reference_texts=int(data["n_reference_texts"]),
            dropped_low_variance=tuple(data["dropped_low_variance"]),
            sd_floor=float(data["sd_floor"]),
        )


def burrows_delta(a: str, b: str, model: StyleModel) -> float:
    """Burrows' Delta between two texts, over the model's function-word features."""
    return model.delta(a, b, family="fw")


def mean_pairwise_delta(
    labelled: Sequence[tuple[str, str]],
    model: StyleModel,
    *,
    cross_author_only: bool = True,
    family_balanced: bool = False,
) -> float:
    """Mean pairwise distance across a set of ``(author_id, text)`` pairs — plan.md §7 M2.

    This is H1's direct measure: run it per round and the prediction is that it shrinks
    monotonically with round number under an unconstrained loop.

    Args:
        labelled: ``(author_id, text)`` for every passage in the round.
        model: A scaler fitted on the *round-0* corpus and never re-fitted (see the module
            docstring — this is the decision that makes or breaks the metric).
        cross_author_only: Compare only passages by different authors. On by default,
            because within-author pairs measure how varied one author's own passages are,
            which is not what "homogenization" names. Turning it off is useful as a
            control: if cross-author distance falls while within-author distance falls
            just as fast, the loop is shortening texts or flattening everything uniformly,
            not converging authors onto one another.
        family_balanced: Use :meth:`StyleModel.family_balanced_delta`.

    Raises:
        MetricError: Fewer than two texts, or no qualifying pair.
    """
    if len(labelled) < 2:
        raise MetricError(f"need at least 2 texts for a pairwise mean, got {len(labelled)}")
    distances = []
    for i in range(len(labelled)):
        for j in range(i + 1, len(labelled)):
            author_a, text_a = labelled[i]
            author_b, text_b = labelled[j]
            if cross_author_only and author_a == author_b:
                continue
            if family_balanced:
                distances.append(model.family_balanced_delta(text_a, text_b))
            else:
                distances.append(model.delta(text_a, text_b))
    if not distances:
        raise MetricError(
            "no qualifying pairs: with cross_author_only=True every text must not share "
            "an author with at least one other. Report this cell as missing rather than 0."
        )
    return mean(distances)


def cluster_purity(labels: Sequence[str], assignments: Sequence[int]) -> float:
    """Fraction of items whose cluster's majority label equals their own — plan.md §4 H3.

    H3 predicts that final-round texts cluster by *reviser model family* rather than by
    source author. That is a comparison of two purities against the same clustering:
    compute it once with author labels and once with reviser labels, and the prediction is
    that the reviser purity is the higher of the two.

    Returns a value in ``[0, 1]``. Note that purity rises mechanically with the number of
    clusters and is 1.0 when every item is its own cluster, so it is only interpretable
    against a fixed clustering — never compare purities computed at different k.

    Raises:
        ValueError: The two sequences differ in length, or are empty.
    """
    if len(labels) != len(assignments):
        raise ValueError(
            f"cluster_purity needs one assignment per label, got {len(labels)} labels and "
            f"{len(assignments)} assignments"
        )
    if not labels:
        raise ValueError("cluster_purity is undefined for an empty set")
    clusters: dict[int, list[str]] = {}
    for label, cluster in zip(labels, assignments, strict=True):
        clusters.setdefault(cluster, []).append(label)
    correct = 0
    for members in clusters.values():
        counts: dict[str, int] = {}
        for label in members:
            counts[label] = counts.get(label, 0) + 1
        correct += max(counts.values())
    return correct / len(labels)


def describe(text: str) -> dict[str, float]:
    """Model-free readouts for one text: the numbers that need no reference corpus.

    Useful for a quick per-round summary and for the Phase-0 degradation curves, which are
    per-passage trajectories rather than cross-passage comparisons.
    """
    out: dict[str, float] = {}
    out.update(sentence_length_stats(text))
    out.update(punctuation_profile(text))
    out["lex:mtld"] = mtld(text)
    out["words"] = float(len(words(text)))
    if math.isnan(out["lex:mtld"]):  # pragma: no cover - mtld raises rather than returning nan
        raise MetricError("MTLD returned NaN")
    return out
