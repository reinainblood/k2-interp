"""Measured interventions and finite-intervention graphs on the original model.

Edges are total effects at the specified intervention dose, with attention free to
change. They are not Anthropic-style direct attribution edges or proof of mediation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from .artifacts import atomic_json, file_hash
from .dictionary import load_dictionary
from .hooks import NativeTrace
from .runtime import token_inputs
from .schema import CaptureSpec, Intervention, Sample, digest, identifier


@torch.no_grad()
def score(
    runtime,
    sample: Sample,
    *,
    patches=None,
    values=None,
    transforms=None,
    observers=None,
    max_context=8192,
):
    prompt, continuation = token_inputs(runtime.tokenizer, sample)
    if not continuation:
        raise ValueError("Causal scoring requires a fixed continuation")
    ids = prompt + continuation[:-1]
    if len(ids) + 1 > max_context:
        raise ValueError("Causal scoring exceeds context bound")
    if max(prompt + continuation) >= runtime.model.config.vocab_size:
        raise ValueError("Token outside vocabulary")
    with NativeTrace(
        runtime.model,
        CaptureSpec(router=False),
        interventions=patches,
        values=values,
        transforms=transforms,
        observers=observers,
    ) as trace:
        trace.begin(list(range(len(ids))), step=0, phase="teacher_forced")
        output = runtime.model(
            input_ids=torch.tensor([ids], device=runtime.device),
            use_cache=False,
            logits_to_keep=len(continuation),
            return_dict=True,
        )
        logits = output.logits[0].float()
        expected = torch.tensor(continuation, device=logits.device)
        logp = logits.log_softmax(-1)
        token_scores = logp.gather(-1, expected[:, None]).squeeze(-1)
        if len(trace.applied) != len(patches or []):
            raise ValueError("An intervention did not fire at the requested layer/position")
    return {
        "sum_logprob": float(token_scores.sum()),
        "mean_logprob": float(token_scores.mean()),
        "token_logprobs": token_scores.tolist(),
        "input_ids_sha256": digest(prompt),
        "continuation_ids": continuation,
    }, logp.cpu()


def kl_from_logp(baseline, edited):
    return float((baseline.exp() * (baseline - edited)).sum(-1).mean())


def private_path(root, path):
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Path escapes artifact root")
    return target


def intervention_sweep(
    runtime,
    run_id: str,
    sample: dict,
    patch_sets: list[list[dict]],
    values_file: str | None = None,
    max_context: int = 8192,
    max_seconds: int = 3600,
):
    sample = Sample.model_validate(sample)
    output = runtime.root / identifier(run_id)
    request = {
        "sample": sample.model_dump(),
        "patch_sets": patch_sets,
        "values_file": values_file,
        "max_context": max_context,
        "identity": runtime.identity,
    }
    values = {}
    if values_file:
        path = private_path(runtime.root, values_file)
        request["values_sha256"] = file_hash(path)
        values = load_file(path)
    identity = digest(request)
    if (output / "request.json").exists():
        if json.loads((output / "request.json").read_text())["sha256"] != identity:
            raise ValueError("Causal run id already belongs to another request")
    atomic_json(output / "request.json", {"sha256": identity, **request})
    baseline, baseline_logp = score(runtime, sample, max_context=max_context)
    atomic_json(output / "baseline.json", baseline)
    runtime.commit()
    deadline = time.monotonic() + max_seconds
    records = []
    for index, patch_set in enumerate(patch_sets):
        record_path = output / f"patch-{index:06d}.json"
        if record_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        if time.monotonic() > deadline:
            raise TimeoutError("Intervention deadline reached; completed patches retained")
        patches = [Intervention.model_validate(p) for p in patch_set]
        metric, logp = score(
            runtime, sample, patches=patches, values=values, max_context=max_context
        )
        record = {
            "index": index,
            "patches": patch_set,
            "metric": metric,
            "delta_sum_logprob": metric["sum_logprob"] - baseline["sum_logprob"],
            "kl_from_baseline": kl_from_logp(baseline_logp, logp),
        }
        atomic_json(record_path, record)
        runtime.commit()
        records.append(record)
    restored, restored_logp = score(runtime, sample, max_context=max_context)
    if not torch.equal(baseline_logp, restored_logp):
        raise ValueError(
            "Unmodified replay changed after interventions; inspect numerical stability/hooks"
        )
    result = {
        "schema": "k2-interp/causal/1",
        "baseline": baseline,
        "interventions": records,
        "cleanup_identity_passed": True,
        "identity": runtime.identity,
        "request_sha256": identity,
    }
    atomic_json(output / "complete.json", result)
    runtime.commit()
    return result


def reconstruction_fidelity(
    runtime, run_id: str, sample: dict, dictionary: str, layer: int, site: str, max_context=8192
):
    """Insert actual reconstructions and measure continuation loss and output KL."""
    sample = Sample.model_validate(sample)
    model, meta = load_dictionary(private_path(runtime.root, dictionary))
    if model.kind == "crosscoder":
        raise ValueError("Joint checkpoint reconstructions require separate per-model evaluation")
    input_view = next(v for v in meta["views"] if v["role"] == "input")
    target_view = next(v for v in meta["views"] if v["role"] == "target")
    input_site = "moe_in" if model.kind == "skip_transcoder" else site
    if (
        target_view["site"] != f"layers.{layer}.{site}"
        or input_view["site"] != f"layers.{layer}.{input_site}"
    ):
        raise ValueError("Dictionary and requested activation sites differ")
    if input_view.get("source_identity", {}).get("revision") != runtime.identity.get("revision"):
        raise ValueError("Dictionary checkpoint differs from target model")
    cached = {}

    def observe(value, positions):
        cached["input"] = value.detach().float()

    def reconstruct(value, positions):
        model.to(value.device)
        x = cached["input"] if model.kind == "skip_transcoder" else value.float()
        pred, _ = model(x.reshape(-1, x.shape[-1]))
        return pred.reshape_as(value).to(value.dtype)

    baseline, logp = score(runtime, sample, max_context=max_context)
    reconstructed, other = score(
        runtime,
        sample,
        max_context=max_context,
        observers={(layer, input_site): observe},
        transforms={(layer, site): reconstruct},
    )
    result = {
        "schema": "k2-interp/fidelity/1",
        "baseline": baseline,
        "reconstructed": reconstructed,
        "delta_mean_nll": baseline["mean_logprob"] - reconstructed["mean_logprob"],
        "kl_from_baseline": kl_from_logp(logp, other),
        "dictionary_sha256": meta["weights_sha256"],
        "layer": layer,
        "site": site,
        "identity": runtime.identity,
    }
    output = runtime.root / identifier(run_id)
    if (output / "fidelity.json").exists():
        raise FileExistsError("Fidelity run already exists")
    atomic_json(output / "fidelity.json", result)
    runtime.commit()
    return result


def attribute_features(runtime, run_id: str, sample: dict, nodes: list[dict], max_context=512):
    """Local gradient x feature attribution to a fixed continuation's log probability.

    This is a first-order estimate on the original model with free attention, not a
    replacement-model CLT graph. Use feature_graph for finite causal verification.
    """
    sample = Sample.model_validate(sample)
    prompt, continuation = token_inputs(runtime.tokenizer, sample)
    if not continuation or len(prompt) + len(continuation) > max_context:
        raise ValueError("Provide a fixed continuation within attribution max_context")
    if not nodes or len({n["id"] for n in nodes}) != len(nodes):
        raise ValueError("Attribution requires unique nodes")
    values = {}
    dictionaries = {}
    for node in nodes:
        model, meta = load_dictionary(private_path(runtime.root, node["dictionary"]))
        if model.kind == "crosscoder":
            raise ValueError("Acausal crosscoders cannot define original-model causal nodes")
        input_site = "moe_in" if model.kind == "skip_transcoder" else node["site"]
        iv = next(v for v in meta["views"] if v["role"] == "input")
        tv = next(v for v in meta["views"] if v["role"] == "target")
        if (
            iv["site"] != f"layers.{node['layer']}.{input_site}"
            or tv["site"] != f"layers.{node['layer']}.{node['site']}"
        ):
            raise ValueError("Attribution dictionary site mismatch")
        if iv.get("source_identity", {}).get("revision") != runtime.identity.get("revision"):
            raise ValueError("Attribution dictionary checkpoint mismatch")
        if (
            not 0 <= node["feature"] < model.width
            or not 0 <= node["position"] < len(prompt) + len(continuation) - 1
        ):
            raise ValueError("Attribution feature/position outside bounds")
        dictionaries[node["id"]] = (model, meta, input_site)

    def remember(key):
        def observer(tensor, positions):
            values[key] = tensor

        return observer

    targets = {(n["layer"], n["site"]) for n in nodes}
    observed = targets | {(n["layer"], dictionaries[n["id"]][2]) for n in nodes}
    transforms = {key: (lambda t, p: t.requires_grad_(True)) for key in targets}
    with (
        torch.enable_grad(),
        NativeTrace(
            runtime.model,
            CaptureSpec(router=False),
            observers={key: remember(key) for key in observed},
            transforms=transforms,
        ) as trace,
    ):
        ids = prompt + continuation[:-1]
        trace.begin(list(range(len(ids))), step=0, phase="teacher_forced")
        out = runtime.model(
            input_ids=torch.tensor([ids], device=runtime.device),
            use_cache=False,
            logits_to_keep=len(continuation),
            return_dict=True,
        )
        logp = out.logits[0].float().log_softmax(-1)
        token_ids = torch.tensor(continuation, device=logp.device)
        metric = logp.gather(-1, token_ids[:, None]).sum()
        ordered = sorted(targets)
        gradients = torch.autograd.grad(metric, [values[k] for k in ordered])
        gradient_map = dict(zip(ordered, gradients, strict=True))
        edges = []
        with torch.no_grad():
            for node in nodes:
                dictionary, meta, input_site = dictionaries[node["id"]]
                x = (
                    values[(node["layer"], input_site)][0, node["position"]]
                    .detach()
                    .float()
                    .cpu()[None]
                )
                activation = dictionary.encode(x)[0, node["feature"]]
                decoder = dictionary.decoder.weight[:, node["feature"]] * dictionary.target_scale
                gradient = (
                    gradient_map[(node["layer"], node["site"])][0, node["position"]].float().cpu()
                )
                sensitivity = float(gradient @ decoder)
                edges.append(
                    {
                        "source": node["id"],
                        "target": "output",
                        "activation": float(activation),
                        "dlogp_dfeature": sensitivity,
                        "attribution": float(activation) * sensitivity,
                        "dictionary_sha256": meta["weights_sha256"],
                    }
                )
        result = {
            "schema": "k2-interp/attribution/1",
            "method": "gradient_times_feature_activation",
            "metric_sum_logprob": float(metric.detach()),
            "nodes": nodes,
            "edges": edges,
            "identity": runtime.identity,
            "input_ids_sha256": digest(ids),
            "limitation": "local first-order effects; no direct feature-to-feature mediation claim",
        }
    output = runtime.root / identifier(run_id)
    if (output / "attribution.json").exists():
        raise FileExistsError("Attribution run already exists")
    atomic_json(output / "attribution.json", result)
    runtime.commit()
    return result


def feature_graph(
    runtime,
    run_id: str,
    sample: dict,
    nodes: list[dict],
    factors: list[float],
    max_context=8192,
    max_seconds=3600,
):
    """nodes: [{id, layer, site, dictionary, feature, position}].

    Each dictionary must reconstruct one site (SAE), or predict moe_out from moe_in
    (skip-transcoder). Checkpoint crosscoders are excluded from causal interventions.
    Residual error is preserved: h' = h + decoder(z' - z). All incoming features
    are observed at their native site after any local decoder-direction intervention.
    """
    sample = Sample.model_validate(sample)
    if not nodes or len({n["id"] for n in nodes}) != len(nodes):
        raise ValueError("Graph requires unique node ids")
    dictionaries = {}
    for node in nodes:
        identifier(node["id"])
        if node["site"] not in ("resid_pre", "resid_mid", "resid_post", "moe_in", "moe_out"):
            raise ValueError("Unsupported dictionary node site")
        if not 0 <= node["layer"] < len(runtime.model.model.layers):
            raise ValueError("Node layer out of range")
        path = private_path(runtime.root, node["dictionary"])
        dictionary, meta = load_dictionary(path)
        if dictionary.kind == "crosscoder":
            raise ValueError("Acausal crosscoders are not directly intervenable model components")
        if len(dictionary.input_dims) != 1 or len(dictionary.target_dims) != 1:
            raise ValueError("Graph requires single-site SAE/transcoder dictionaries")
        if not 0 <= node["feature"] < dictionary.width:
            raise ValueError("Feature id out of range")
        if dictionary.kind == "skip_transcoder" and node["site"] != "moe_out":
            raise ValueError("Skip-transcoder graph nodes must write at moe_out")
        view_input = next(v for v in meta["views"] if v["role"] == "input")
        view_target = next(v for v in meta["views"] if v["role"] == "target")
        expected_input = "moe_in" if dictionary.kind == "skip_transcoder" else node["site"]
        if (
            view_input["site"] != f"layers.{node['layer']}.{expected_input}"
            or view_target["site"] != f"layers.{node['layer']}.{node['site']}"
        ):
            raise ValueError("Dictionary site does not match graph node")
        source_identity = view_input.get("source_identity", {})
        if source_identity.get("revision") != runtime.identity.get("revision"):
            raise ValueError("Dictionary checkpoint differs from target model")
        dictionaries[node["id"]] = (dictionary, meta)

    def measure(source=None, factor=1.0):
        seen, inputs = {}, {}
        observers, transforms = {}, {}

        def make_observer(layer, site):
            def observer(value, positions):
                for node in nodes:
                    dictionary, _ = dictionaries[node["id"]]
                    input_site = "moe_in" if dictionary.kind == "skip_transcoder" else node["site"]
                    if node["layer"] != layer or input_site != site:
                        continue
                    if node["position"] not in positions:
                        raise ValueError("Feature position not processed by scoring input")
                    row = value[0, positions.index(node["position"])].float().unsqueeze(0)
                    dictionary.to(row.device)
                    inputs[node["id"]] = row
                    seen[node["id"]] = float(dictionary.encode(row)[0, node["feature"]])

            return observer

        for node in nodes:
            dictionary, _ = dictionaries[node["id"]]
            key = (
                node["layer"],
                "moe_in" if dictionary.kind == "skip_transcoder" else node["site"],
            )
            observers[key] = make_observer(*key)
        if source:
            dictionary, _ = dictionaries[source["id"]]

            def transform(value, positions):
                # For an SAE the current value is also its encoder input.
                row_idx = positions.index(source["position"])
                x = (
                    inputs[source["id"]]
                    if dictionary.kind == "skip_transcoder"
                    else value[0, row_idx].float().unsqueeze(0)
                )
                dictionary.to(value.device)
                activation = dictionary.encode(x)[0, source["feature"]]
                operation = source.get("operation", "scale")
                if operation not in ("scale", "set", "add"):
                    raise ValueError("Feature operation must be scale, set, or add")
                change = (
                    (factor - 1) * activation
                    if operation == "scale"
                    else (factor - activation if operation == "set" else factor)
                )
                delta = (
                    change
                    * dictionary.decoder.weight[:, source["feature"]]
                    * dictionary.target_scale
                )
                edited = value.clone()
                edited[0, row_idx] += delta.to(value.dtype)
                return edited

            transforms[(source["layer"], source["site"])] = transform
        metric, logp = score(
            runtime, sample, observers=observers, transforms=transforms, max_context=max_context
        )
        if set(seen) != {n["id"] for n in nodes}:
            raise ValueError("Some feature graph nodes were not observed")
        return metric, logp, seen

    output = runtime.root / identifier(run_id)
    request = {
        "sample": sample.model_dump(),
        "nodes": nodes,
        "factors": factors,
        "identity": runtime.identity,
        "dictionary_hashes": {k: v[1]["weights_sha256"] for k, v in dictionaries.items()},
    }
    if (output / "request.json").exists() and json.loads(
        (output / "request.json").read_text()
    ) != request:
        raise ValueError("Graph run id already exists with another request")
    atomic_json(output / "request.json", request)
    baseline, base_logp, base_features = measure()
    deadline = time.monotonic() + max_seconds
    interventions, edges = [], []
    for i, source in enumerate(nodes):
        for j, factor in enumerate(factors):
            if time.monotonic() > deadline:
                raise TimeoutError("Graph time limit reached")
            patch_path = output / f"node-{i:04d}-dose-{j:03d}.json"
            if patch_path.exists():
                result = json.loads(patch_path.read_text())
            else:
                metric, logp, features = measure(source, factor)
                result = {
                    "source": source["id"],
                    "factor": factor,
                    "features": features,
                    "metric": metric,
                    "kl_from_baseline": kl_from_logp(base_logp, logp),
                }
                atomic_json(patch_path, result)
                runtime.commit()
            interventions.append(result)
            for target in nodes:
                if target["id"] != source["id"]:
                    edges.append(
                        {
                            "source": source["id"],
                            "target": target["id"],
                            "factor": factor,
                            "effect": result["features"][target["id"]]
                            - base_features[target["id"]],
                            "type": "measured_total_effect_on_feature",
                        }
                    )
            edges.append(
                {
                    "source": source["id"],
                    "target": "output",
                    "factor": factor,
                    "effect": result["metric"]["sum_logprob"] - baseline["sum_logprob"],
                    "type": "measured_total_effect_on_logprob",
                }
            )
    _, restored, _ = measure()
    if not torch.equal(base_logp, restored):
        raise ValueError("Graph interventions failed cleanup identity check")
    result = {
        "schema": "k2-interp/graph/1",
        "method": "finite-intervention total-effect graph",
        "direct_edges": False,
        "attention_frozen": False,
        "nodes": nodes,
        "edges": edges,
        "baseline": baseline,
        "baseline_features": base_features,
        "interventions": interventions,
        "cleanup_identity_passed": True,
        "request_sha256": digest(request),
        "identity": runtime.identity,
    }
    atomic_json(output / "graph.json", result)
    runtime.commit()
    return result
