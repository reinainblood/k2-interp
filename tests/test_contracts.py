import pytest

from k2_interp.cli import panel, parser
from k2_interp.schema import CaptureSpec, Intervention, RunSpec, Sample


def test_contracts_reject_silent_mistakes():
    with pytest.raises(ValueError):
        Sample(id="../escape", text="test")
    with pytest.raises(ValueError):
        Sample(id="x", text="test", messages=[])
    with pytest.raises(ValueError):
        RunSpec(run_id="x", samples=[Sample(id="s", text="a"), Sample(id="s", text="b")])
    with pytest.raises(ValueError):
        CaptureSpec(sites=["residual-ish"])
    with pytest.raises(ValueError):
        Intervention(layer=3, site="expert_ids", positions=[0], operation="scale")
    assert len(panel(["all"])) == 11
    assert len(panel(["spine"])) == 6
    assert parser().parse_args(["deploy", "--checkpoints", "all"]).command == "deploy"


def test_reasoning_and_eos_parsing():
    from types import SimpleNamespace

    from k2_interp.runtime import response_message

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(
                {1: "<eos>", 2: "<think>", 3: "reasoning", 4: "</think>", 5: "answer", 11: "word1"}[
                    i
                ]
                for i in ids
            )

    tok = Tokenizer()
    cfg = SimpleNamespace(eos_token_id=1)
    assert response_message(tok, cfg, [2], [3, 4, 5, 1]) == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reasoning",
    }
    assert response_message(tok, cfg, [2], [3]) == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reasoning",
    }
    assert response_message(tok, cfg, [5], [11])["content"] == "word1"


def test_actual_ifm_template_roundtrip(native_tokenizer):
    from types import SimpleNamespace

    from k2_interp.runtime import response_message

    tok = native_tokenizer
    for effort, tag, field in (
        ("high", "ifm|think", "reasoning_content"),
        ("medium", "ifm|think_fast", "think_fast"),
        ("low", "ifm|think_faster", "think_faster"),
    ):
        messages = [{"role": "user", "content": "Name a city."}]
        prompt = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=effort
        )
        answer = tok.encode(f"Consider Paris.</{tag}>Paris", add_special_tokens=False)
        message = response_message(
            tok, SimpleNamespace(eos_token_id=tok.eos_token_id), prompt, answer
        )
        assert message["content"] == "Paris" and message[field] == "Consider Paris."
        rendered = tok.apply_chat_template(
            messages + [message, {"role": "user", "content": "Another?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert f"<{tag}>" in rendered and "Consider Paris." in rendered
        incomplete = response_message(
            tok,
            SimpleNamespace(eos_token_id=tok.eos_token_id),
            prompt,
            tok.encode("Thinking", add_special_tokens=False),
        )
        assert incomplete["content"] == "" and incomplete["reasoning_content"] == "Thinking"
    empty = {"role": "assistant", "content": "Paris", "reasoning_content": ""}
    tok.apply_chat_template(messages + [empty], tokenize=True)
