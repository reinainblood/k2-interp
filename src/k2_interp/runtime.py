"""Chat, teacher-forced scoring, and bounded trace execution on the same model."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

import torch

from .artifacts import TraceWriter, file_hash
from .hooks import NativeTrace
from .schema import CaptureSpec, GenerationSpec, RunSpec, Sample, digest


def token_inputs(tokenizer, sample: Sample) -> tuple[list[int], list[int]]:
    if sample.input_ids is not None:
        prompt = sample.input_ids.copy()
    elif sample.messages is not None:
        prompt = tokenizer.apply_chat_template(
            sample.messages,
            tokenize=True,
            add_generation_prompt=True,
            **sample.chat_template_kwargs,
        )
    else:
        prompt = tokenizer.encode(sample.text, add_special_tokens=True)
    continuation = sample.continuation_ids or (
        tokenizer.encode(sample.continuation, add_special_tokens=False)
        if sample.continuation is not None
        else []
    )
    if not prompt:
        raise ValueError("Tokenized prompt is empty")
    return list(prompt), list(continuation)


def sample_token(logits: torch.Tensor, spec: GenerationSpec, generator) -> int:
    if spec.temperature == 0:
        return int(logits.argmax(-1).item())
    scores = logits.float() / spec.temperature
    if spec.top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(-1).cumsum(-1)
        remove = cumulative - sorted_scores.softmax(-1) >= spec.top_p
        sorted_scores[remove] = -torch.inf
        scores = torch.empty_like(scores).scatter(0, order, sorted_scores)
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())


def response_message(tokenizer, generation_config, prompt: list[int], generated: list[int]) -> dict:
    """Parse native thinking delimiters while preserving exact raw output elsewhere."""
    eos = generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    content_ids = list(generated)
    while content_ids and content_ids[-1] in eos_ids:
        content_ids.pop()
    text = tokenizer.decode(content_ids, skip_special_tokens=False)
    reasoning, selected_tag = "", "ifm|think"
    prompt_text = tokenizer.decode(prompt, skip_special_tokens=False).rstrip()
    for tag in ("ifm|think", "ifm|think_fast", "ifm|think_faster", "think"):
        opening, closing = f"<{tag}>", f"</{tag}>"
        if closing in text:
            reasoning, text = text.split(closing, 1)
            reasoning = reasoning.removeprefix(opening).strip()
            selected_tag = tag
            break
        if text.lstrip().startswith(opening) or prompt_text.endswith(opening):
            reasoning, text = text.removeprefix(opening).strip(), ""
            selected_tag = tag
            break
    # K2 requires an explicit thinking field on every assistant history turn,
    # including an empty field when there was no generated reasoning.
    message = {"role": "assistant", "content": text.strip(), "reasoning_content": reasoning}
    if selected_tag in ("ifm|think_fast", "ifm|think_faster"):
        message[selected_tag.removeprefix("ifm|")] = reasoning
    return message


class Runtime:
    def __init__(self, model, tokenizer, identity: dict, root: str | Path, commit=lambda: None):
        self.model, self.tokenizer, self.identity = model, tokenizer, dict(identity)
        self.root, self.commit = Path(root), commit
        self.identity["runtime_source_sha256"] = digest(
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        )
        self.device = model.get_input_embeddings().weight.device
        self.identity["packages"] = {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "accelerate", "safetensors", "k2-interp")
        }

    def run(self, spec: RunSpec) -> dict:
        if spec.checkpoint != self.identity["checkpoint"]:
            raise ValueError("Request checkpoint does not match worker checkpoint")
        writer = TraceWriter(self.root, spec.model_dump(), self.identity, self.commit)
        receipts = {}
        deadline = time.monotonic() + spec.max_seconds
        for sample in spec.samples:
            if receipt := writer.completed(sample.id):
                receipts[sample.id] = {**receipt, "resumed": True}
                continue
            sink = writer.sample(sample.id, spec.capture.max_bytes)
            try:
                with NativeTrace(self.model, spec.capture, sink) as trace:
                    result = self.execute(
                        sample, spec.generation, trace, spec.max_context, deadline
                    )
                result["sample"] = sample.model_dump()
                result["input_sha256"] = digest(result["input_ids"])
                receipts[sample.id] = sink.finish(result)
            except Exception as exc:
                sink.fail(exc)
                raise
        return {
            "run_id": spec.run_id,
            "checkpoint": spec.checkpoint,
            "path": str(writer.root),
            "samples": receipts,
            "status": "complete",
        }

    @torch.no_grad()
    def execute(
        self,
        sample: Sample,
        generation: GenerationSpec,
        trace: NativeTrace,
        max_context: int,
        deadline: float,
    ) -> dict:
        prompt, continuation = token_inputs(self.tokenizer, sample)
        extra = len(continuation) if continuation else generation.max_new_tokens
        if len(prompt) + extra > max_context:
            raise ValueError(
                "Prompt plus output exceeds request context bound; no truncation applied"
            )
        if max(prompt + continuation) >= self.model.config.vocab_size:
            raise ValueError("Input token exceeds vocabulary")
        generator = None
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        all_ids, generated, scores = list(prompt), [], []
        cached = None
        forward_ids = prompt
        start = 0
        step = 0
        # One final pass processes the last generated/teacher-forced token as well.
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Run wall-time limit reached; partial trace retained")
            phase = "prefill" if step == 0 else ("teacher_forced" if continuation else "decode")
            trace.begin(list(range(start, start + len(forward_ids))), step=step, phase=phase)
            x = torch.tensor([forward_ids], device=self.device)
            mask = torch.ones((1, len(all_ids)), dtype=torch.long, device=self.device)
            out = self.model(
                input_ids=x,
                attention_mask=mask,
                past_key_values=cached,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            cached = out.past_key_values
            logits = out.logits[0, -1].float()
            if generator is None:
                generator = torch.Generator(device=logits.device).manual_seed(generation.seed)
            if trace.sink:
                # Per-forward persistence also survives interruption during a long response.
                trace.flush()
                trace.sink.checkpoint()
            if len(generated) == extra or (generated and not continuation and generated[-1] in eos):
                break
            next_token = (
                continuation[len(generated)]
                if continuation
                else sample_token(logits, generation, generator)
            )
            scores.append(float(logits.log_softmax(-1)[next_token]))
            generated.append(next_token)
            start = len(all_ids)
            all_ids.append(next_token)
            forward_ids = [next_token]
            step += 1
        return {
            "input_ids": prompt,
            "output_ids": generated,
            "all_token_ids": all_ids,
            "token_text": self.tokenizer.convert_ids_to_tokens(all_ids),
            "output_text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "assistant_message": response_message(
                self.tokenizer, self.model.generation_config, prompt, generated
            ),
            "output_logprobs": scores,
            "continuation_logprob_sum": sum(scores) if continuation else None,
            "continuation_logprob_mean": sum(scores) / len(scores) if continuation else None,
            "mode": "teacher_forced" if continuation else "generation",
            "finish_reason": "eos" if generated and generated[-1] in eos else "length",
            "capture_alignment": "row at token position i is the activation processing token i, predicting i+1",
        }

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens=256,
        seed=0,
        temperature=0,
        top_p=1,
        chat_template_kwargs=None,
        max_context=8192,
        capture: CaptureSpec | None = None,
        run_id: str,
    ):
        spec = RunSpec(
            run_id=run_id,
            checkpoint=self.identity["checkpoint"],
            samples=[
                Sample(
                    id="chat",
                    messages=messages,
                    chat_template_kwargs=chat_template_kwargs or {"reasoning_effort": "high"},
                )
            ],
            generation=GenerationSpec(
                max_new_tokens=max_new_tokens, seed=seed, temperature=temperature, top_p=top_p
            ),
            capture=capture or CaptureSpec(router=False, layers=[]),
            max_context=max_context,
        )
        receipt = self.run(spec)
        sample = receipt["samples"]["chat"]
        result = json.loads((self.root / run_id / sample["attempt"] / "result.json").read_text())
        message = result["assistant_message"]
        return {"result": result, "receipt": receipt, "assistant_message": message}
