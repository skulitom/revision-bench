"""Local model access, via Ollama on ``http://localhost:11434``.

This is the only module in the repo that talks to a model, and it exists to make three
specific silent failures impossible.

**Silent context truncation.** Ollama applies a default context window (``num_ctx``) and
quietly drops whatever does not fit. A 900-word passage plus its instruction is ~1300
tokens, so a default that is too small would hand the model *part* of the passage. The
model would then dutifully return a revision of that part, and the round would look like a
reviser that deleted half the text — which is exactly the shape of the confound
``docs/findings-phase0.md`` §5.2 is already about, but caused by us rather than by it.
``num_ctx`` is therefore required, never defaulted, and :meth:`OllamaClient.generate`
checks the realised token counts against it afterwards.

**Silent output truncation.** When generation hits ``num_predict`` the reply is cut
mid-sentence and ``done_reason`` comes back as ``"length"`` rather than ``"stop"``. Cut
text scores as a shorter, flatter passage. The reason is recorded on every generation and
the runner treats anything other than ``"stop"`` as a flagged round.

**The cold-start round.** Measured on this machine (findings §5.1): the first generation
after a model is loaded into VRAM differs from every later one, and all later ones are
byte-identical to each other. It holds for greedy decoding *and* for seeded sampling, so it
is not a sampler property — it is the load path. Five calls at fixed settings:

    greedy      cold 276 words, then 291 / 291 / 291 / 291   (identical)
    temp 0.8    cold 286 words, then 384 / 384 / 384 / 384   (identical)

Left alone this is nasty rather than merely untidy. Ollama unloads an idle model after
``keep_alive`` (5 minutes by default), so a long sweep with any pause in it acquires a
scattering of cold rounds — irreproducible, and indistinguishable in the artifact from
genuine loop dynamics. :class:`OllamaClient` therefore pins ``keep_alive`` on every
request and offers :meth:`OllamaClient.warm_up`, which the runner calls before the first
scored generation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

__all__ = [
    "DEFAULT_HOST",
    "Generation",
    "GenerationOptions",
    "ModelIdentity",
    "OllamaClient",
    "OllamaError",
]

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Ollama is unreachable, the model is absent, or a generation came back unusable."""


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Sampling and context settings, pinned rather than defaulted.

    Every field is required. Ollama's own defaults are version-dependent and invisible in
    a result file, and a benchmark whose numbers move when the runtime changes its default
    context length is not a benchmark.
    """

    seed: int
    temperature: float
    top_k: int
    top_p: float
    num_ctx: int
    num_predict: int
    repeat_penalty: float

    def __post_init__(self) -> None:
        # Note what is deliberately NOT checked here: there is no rule tying temperature to
        # top_k. An earlier version of this class refused `temperature=0` with `top_k != 1`,
        # on the strength of a probe that appeared to show greedy decoding was the only
        # reproducible setting. That was wrong — the probe's "deterministic" calls simply
        # all happened to be warm. Both samplers reproduce once warm and neither does cold
        # (see the module docstring), so the guard was enforcing a superstition. Warmth is
        # the variable that matters, and it is not a property of this dataclass.
        if self.num_ctx < 512:
            raise ValueError(f"num_ctx={self.num_ctx} is too small to hold a passage")
        if self.num_predict < 1:
            raise ValueError(
                f"num_predict={self.num_predict} must be a positive cap; -1 (unbounded) is "
                "refused because a runaway generation would stall a 100-round sweep"
            )

    def as_ollama_options(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_deterministic(self) -> bool:
        """True when repeated calls are expected to reproduce byte-identically.

        Both greedy decoding and seeded sampling qualify: what breaks reproducibility on
        this stack is a cold model, not the sampler. Sampling without a pinned seed does
        not qualify, and Ollama treats a negative seed as "choose one".
        """
        return self.seed >= 0


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Everything needed to say which weights produced a number.

    The **digest** is the load-bearing field, not the tag. A tag is a mutable pointer: a
    later ``ollama pull`` can move ``gemma3:4b`` to different weights, and a result file
    naming only the tag would then be untraceable. plan.md §12.5 was retired on the
    strength of this field existing.
    """

    tag: str
    digest: str
    family: str
    parameter_size: str
    quantization_level: str
    ollama_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.tag} ({self.family} {self.parameter_size} {self.quantization_level}, "
            f"digest {self.digest[:12]}, ollama {self.ollama_version})"
        )


@dataclass(frozen=True, slots=True)
class Generation:
    """One completion plus the facts needed to distrust it."""

    text: str
    prompt_tokens: int
    output_tokens: int
    #: "stop" for a natural end; "length" means num_predict cut it mid-sentence.
    done_reason: str
    wall_seconds: float

    @property
    def truncated(self) -> bool:
        return self.done_reason != "stop"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")  # the caller stores the text under its own key
        data["truncated"] = self.truncated
        return data


class OllamaClient:
    """Thin, synchronous client. One model call at a time, by design.

    There is a single GPU. Concurrency here would only produce contention and would make
    wall-clock numbers meaningless, so this does not batch or parallelise.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        timeout: float = 900.0,
        keep_alive: str = "60m",
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        #: Sent on every request. Ollama's 5-minute default would unload the model during
        #: any pause in a sweep, and the reload's first generation is the irreproducible
        #: one (see the module docstring).
        self.keep_alive = keep_alive
        self._session = requests.Session()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}{path}"
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc}). Start it with `ollama serve`, "
                f"or point --host somewhere else."
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaError(
                f"{url} returned {response.status_code}: {response.text[:400]}"
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"{url} failed: {exc}") from exc
        return response.json()

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.host}{path}"
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        return response.json()

    def identity(self, tag: str) -> ModelIdentity:
        """Resolve a model tag to its digest and details.

        Raises:
            OllamaError: The tag is not installed. The message lists what is, because the
                usual cause is a typo or a model that was never pulled, and a run that
                fell back to some other model would be worse than one that stopped.
        """
        version = str(self._get("/api/version").get("version", "unknown"))
        listing = self._get("/api/tags")
        models = listing.get("models", [])
        for entry in models:
            if entry.get("name") == tag:
                details = entry.get("details", {})
                return ModelIdentity(
                    tag=tag,
                    digest=str(entry.get("digest", "")),
                    family=str(details.get("family", "")),
                    parameter_size=str(details.get("parameter_size", "")),
                    quantization_level=str(details.get("quantization_level", "")),
                    ollama_version=version,
                )
        installed = ", ".join(sorted(str(m.get("name")) for m in models)) or "(none)"
        raise OllamaError(f"model {tag!r} is not installed. Installed: {installed}")

    def warm_up(self, tag: str, options: GenerationOptions) -> Generation:
        """Run and discard one generation, so the next call is a warm one.

        The first generation after a load is irreproducible (module docstring), so the
        runner spends one throwaway call to move that cost off the first *scored* round.
        The prompt is deliberately unrelated to the corpus: it must not prime anything, and
        it should be short enough to be cheap.

        Returns the discarded generation, so a caller can log what it cost.
        """
        probe = GenerationOptions(
            seed=options.seed,
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            num_ctx=options.num_ctx,
            num_predict=8,
            repeat_penalty=options.repeat_penalty,
        )
        return self.generate(tag, "Say the single word: ready.", probe)

    def generate(self, tag: str, prompt: str, options: GenerationOptions) -> Generation:
        """Run one completion.

        Raises:
            OllamaError: The call failed, the reply was empty, or the prompt did not fit
                in ``num_ctx``. The last is fatal rather than a warning: a truncated prompt
                means the model never saw part of the passage, and every metric computed
                from the result would be measuring our bug.
        """
        started = time.time()
        payload = {
            "model": tag,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options.as_ollama_options(),
        }
        data = self._post("/api/generate", payload)
        elapsed = time.time() - started

        text = data.get("response", "")
        if not text.strip():
            raise OllamaError(
                f"{tag} returned an empty completion (done_reason="
                f"{data.get('done_reason')!r}); this is not a revision and must not be "
                f"scored as one"
            )

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)

        # The prompt-fit check. Ollama does not report that it truncated, so infer it: a
        # prompt occupying essentially the whole window means the tail was dropped.
        if prompt_tokens >= options.num_ctx:
            raise OllamaError(
                f"prompt used {prompt_tokens} tokens against num_ctx={options.num_ctx}, so "
                f"Ollama silently dropped the end of the passage. Raise num_ctx. Scoring "
                f"this round would measure our truncation, not the model's revision."
            )

        return Generation(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            done_reason=str(data.get("done_reason") or "unknown"),
            wall_seconds=round(elapsed, 2),
        )
