"""The Ollama client's guards. Offline: the HTTP layer is stubbed."""

from __future__ import annotations

import pytest

from revisionbench.ollama import (
    Generation,
    GenerationOptions,
    ModelIdentity,
    OllamaClient,
    OllamaError,
)

BASE = {
    "seed": 0,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "num_ctx": 8192,
    "num_predict": 3072,
    "repeat_penalty": 1.0,
}


class TestGenerationOptions:
    def test_round_trips_to_ollama_options(self) -> None:
        assert GenerationOptions(**BASE).as_ollama_options() == BASE

    def test_greedy_is_allowed(self) -> None:
        """An earlier version banned temperature=0 with top_k!=1 on a mistaken premise.

        The probe behind that rule had accidentally compared warm calls with a cold one;
        both samplers reproduce once warm and neither does cold, so tying temperature to
        top_k was enforcing a superstition. Pinned so it does not come back.
        """
        options = GenerationOptions(**{**BASE, "temperature": 0.0, "top_k": 40})
        assert options.is_deterministic

    def test_determinism_is_about_the_seed(self) -> None:
        assert GenerationOptions(**BASE).is_deterministic is True
        assert GenerationOptions(**{**BASE, "seed": -1}).is_deterministic is False

    def test_rejects_a_context_too_small_for_a_passage(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            GenerationOptions(**{**BASE, "num_ctx": 128})

    def test_rejects_unbounded_generation(self) -> None:
        with pytest.raises(ValueError, match="positive cap"):
            GenerationOptions(**{**BASE, "num_predict": -1})


class TestGeneration:
    def test_truncation_is_derived_from_done_reason(self) -> None:
        stopped = Generation("x", 10, 5, "stop", 1.0)
        cut = Generation("x", 10, 5, "length", 1.0)
        assert stopped.truncated is False
        assert cut.truncated is True

    def test_as_dict_omits_the_text(self) -> None:
        """The caller stores text under its own key; duplicating it doubles artifact size."""
        data = Generation("some prose", 10, 5, "stop", 1.0).as_dict()
        assert "text" not in data
        assert data["truncated"] is False


class StubClient(OllamaClient):
    """An OllamaClient whose HTTP calls are replaced by canned payloads."""

    def __init__(self, payload: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.payload = payload
        self.last_request: dict | None = None

    def _post(self, path: str, payload: dict):  # type: ignore[override]
        self.last_request = payload
        return self.payload


class TestGenerateGuards:
    def make(self, **overrides):
        payload = {
            "response": "A revised passage.",
            "prompt_eval_count": 1300,
            "eval_count": 500,
            "done_reason": "stop",
        }
        payload.update(overrides)
        return StubClient(payload)

    def test_happy_path(self) -> None:
        client = self.make()
        result = client.generate("m", "prompt", GenerationOptions(**BASE))
        assert result.text == "A revised passage."
        assert result.prompt_tokens == 1300
        assert result.truncated is False

    def test_keep_alive_is_sent_on_every_request(self) -> None:
        """Ollama's 5-minute default would unload the model mid-sweep, and the reload's
        first generation is the irreproducible one."""
        client = self.make(**{})
        client.keep_alive = "60m"
        client.generate("m", "prompt", GenerationOptions(**BASE))
        assert client.last_request["keep_alive"] == "60m"

    def test_empty_completion_is_fatal(self) -> None:
        client = self.make(response="   \n ")
        with pytest.raises(OllamaError, match="empty completion"):
            client.generate("m", "prompt", GenerationOptions(**BASE))

    def test_prompt_filling_the_context_is_fatal(self) -> None:
        """Ollama does not report truncation, so a full window means the tail was dropped.

        Scoring that round would measure our bug as if it were the model deleting text --
        and 'the reviser cut the passage' is exactly the finding this project is looking
        for, which is what makes it dangerous.
        """
        client = self.make(prompt_eval_count=8192)
        with pytest.raises(OllamaError, match="silently dropped"):
            client.generate("m", "prompt", GenerationOptions(**BASE))

    def test_truncated_output_is_recorded_not_raised(self) -> None:
        """A cut generation is a flagged round, not a crash; the runner records it."""
        client = self.make(done_reason="length")
        assert client.generate("m", "p", GenerationOptions(**BASE)).truncated is True


class TestIdentity:
    class TagsStub(OllamaClient):
        def _get(self, path: str):  # type: ignore[override]
            if path == "/api/version":
                return {"version": "0.32.8"}
            return {
                "models": [
                    {
                        "name": "gemma3:4b",
                        "digest": "a" * 64,
                        "details": {
                            "family": "gemma3",
                            "parameter_size": "4.3B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            }

    def test_resolves_digest_and_details(self) -> None:
        identity = self.TagsStub().identity("gemma3:4b")
        assert identity.digest == "a" * 64
        assert identity.family == "gemma3"
        assert identity.ollama_version == "0.32.8"
        assert "gemma3:4b" in identity.summary()

    def test_missing_model_lists_what_is_installed(self) -> None:
        with pytest.raises(OllamaError, match="Installed: gemma3:4b"):
            self.TagsStub().identity("llama3.2:1b")


def test_model_identity_summary_leads_with_the_digest_prefix() -> None:
    identity = ModelIdentity("t", "b" * 64, "fam", "4B", "Q4", "0.1")
    assert "digest bbbbbbbbbbbb" in identity.summary()
