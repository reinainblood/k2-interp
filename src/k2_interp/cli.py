"""Explicit CLI: importing or listing checkpoints never starts GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .artifacts import atomic_json
from .registry import CHECKPOINTS, SPINE, resolve
from .schema import RunSpec


def app_name(checkpoint):
    if checkpoint not in CHECKPOINTS:
        raise ValueError("Unknown checkpoint")
    return "k2-interp-" + checkpoint.replace("_", "-")


def remote(checkpoint):
    import modal

    modal.Function.from_name(app_name(checkpoint), "preflight", environment_name="FDR").remote()
    return modal.Cls.from_name(app_name(checkpoint), "NativeWorker", environment_name="FDR")()


def json_file(path):
    return json.loads(Path(path).read_text())


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(prog="k2-interp", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("checkpoints", help="List pinned checkpoint commits; no network or GPU")
    lock = sub.add_parser("lock", help="Resolve current Hub metadata at pinned commits")
    lock.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
    lock.add_argument("--output", required=True)
    deploy = sub.add_parser("deploy", help="Register scale-to-zero FDR checkpoint apps")
    deploy.add_argument("--checkpoints", nargs="+", default=["sft_3_1500"])
    deploy.add_argument("--idle-seconds", type=int, default=300)
    prepare = sub.add_parser("prepare", help="Download/hash checkpoint on CPU")
    prepare.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
    prepare.add_argument("--reuse-subdir", default="")
    prepare.add_argument("--detach", action="store_true")
    run = sub.add_parser("run", help="Run an evaluation/capture request on a prepared checkpoint")
    run.add_argument("request")
    run.add_argument("--detach", action="store_true")
    batch = sub.add_parser("batch", help="Run the same protocol over a checkpoint panel")
    batch.add_argument("request")
    batch.add_argument("--checkpoints", nargs="+", default=["spine"])
    batch.add_argument("--parallel", type=int, default=1)
    call = sub.add_parser("call", help="Read a detached Modal call result")
    call.add_argument("id")
    call.add_argument("--timeout", type=int, default=0)
    chat = sub.add_parser("chat", help="Interactive private chat; each turn has a trace receipt")
    chat.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
    chat.add_argument("--system-file")
    chat.add_argument("--max-new-tokens", type=int, default=512)
    smoke = sub.add_parser(
        "smoke", help="Run bounded full-model parity/capture/intervention checks"
    )
    smoke.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
    smoke.add_argument("--run-id", default="smoke-" + uuid.uuid4().hex[:12])
    smoke.add_argument("--detach", action="store_true")
    for name in ("intervene", "graph", "attribute", "fidelity"):
        c = sub.add_parser(name, help="Run a causal/attribution request on the original model")
        c.add_argument("request")
        c.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
        c.add_argument("--detach", action="store_true")
    dataset = sub.add_parser("dataset", help="Build an aligned activation dataset locally")
    dataset.add_argument("views", help="JSON list of {name, run, site, role}")
    dataset.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Train SAE, skip-transcoder, or checkpoint crosscoder")
    train.add_argument("config")
    train.add_argument("--remote", action="store_true")
    train.add_argument("--checkpoint", choices=CHECKPOINTS, default="sft_3_1500")
    train.add_argument("--detach", action="store_true")
    loc = sub.add_parser("localize", help="Paired residual localization with grouped bootstrap")
    loc.add_argument("run")
    loc.add_argument("pairs")
    loc.add_argument("--sites", nargs="+", required=True)
    loc.add_argument("--output", required=True)
    routes = sub.add_parser("compare-routes", help="Compare aligned router traces")
    routes.add_argument("run_a")
    routes.add_argument("run_b")
    routes.add_argument("--layers", nargs="+", type=int, required=True)
    routes.add_argument("--output", required=True)
    features = sub.add_parser(
        "features", help="Summarize dictionary features on held-out activations"
    )
    features.add_argument("dictionary")
    features.add_argument("dataset")
    features.add_argument("--output", required=True)
    sub.add_parser("status", help="List only this runtime's FDR apps without loading models")
    return p


def panel(names):
    if names == ["all"]:
        return list(CHECKPOINTS)
    if names == ["spine"]:
        return list(SPINE)
    if not names or any(n not in CHECKPOINTS for n in names) or len(set(names)) != len(names):
        raise ValueError("Select unique checkpoint tags, or all/spine")
    return names


def invoke(method, args, detach):
    if detach:
        call = method.spawn(*args)
        return {"call_id": call.object_id, "state": "submitted"}
    return method.remote(*args)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "checkpoints":
        emit(CHECKPOINTS)
    elif args.command == "lock":
        lock = resolve(args.checkpoint)
        atomic_json(Path(args.output), lock)
        emit({k: v for k, v in lock.items() if k != "files"})
    elif args.command == "deploy":
        if not 2 <= args.idle_seconds <= 3600:
            raise ValueError("idle-seconds must be between 2 and 3600")
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
            raise ValueError("Commit the source before deploying so run provenance is reproducible")
        for checkpoint in panel(args.checkpoints):
            env = {
                **os.environ,
                "K2_CHECKPOINT": checkpoint,
                "K2_INTERP_CODE_REVISION": revision,
                "K2_IDLE_SECONDS": str(args.idle_seconds),
            }
            subprocess.run(
                [sys.executable, "-m", "modal", "deploy", "--env=FDR", "-m", "k2_interp.modal_app"],
                env=env,
                cwd=root,
                check=True,
            )
    elif args.command == "prepare":
        import modal

        method = modal.Function.from_name(
            app_name(args.checkpoint), "prepare_checkpoint", environment_name="FDR"
        )
        result = invoke(method, [args.reuse_subdir], args.detach)
        emit({k: v for k, v in result.items() if k not in ("files", "local_files")})
    elif args.command == "run":
        spec = RunSpec.model_validate(json_file(args.request))
        emit(invoke(remote(spec.checkpoint).run, [spec.model_dump()], args.detach))
    elif args.command == "batch":
        checkpoints = panel(args.checkpoints)
        if not 1 <= args.parallel <= len(checkpoints):
            raise ValueError("parallel must be within the checkpoint panel size")
        spec = RunSpec.model_validate(json_file(args.request))

        def run(checkpoint):
            request = spec.model_copy(update={"checkpoint": checkpoint})
            return remote(checkpoint).run.remote(request.model_dump())

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            pending = {pool.submit(run, c): c for c in checkpoints}
            for future in concurrent.futures.as_completed(pending):
                emit({"checkpoint": pending[future], "result": future.result()})
    elif args.command == "call":
        import modal

        try:
            emit(modal.FunctionCall.from_id(args.id).get(timeout=args.timeout))
        except TimeoutError:
            emit({"call_id": args.id, "state": "running"})
    elif args.command == "chat":
        worker = remote(args.checkpoint)
        messages = []
        if args.system_file:
            messages.append({"role": "system", "content": Path(args.system_file).read_text()})
        while True:
            try:
                text = input("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text == ":quit":
                break
            messages.append({"role": "user", "content": text})
            result = worker.chat.remote(
                {
                    "messages": messages,
                    "run_id": "chat-" + uuid.uuid4().hex,
                    "max_new_tokens": args.max_new_tokens,
                }
            )
            message = result["assistant_message"]
            print("model>", message["content"])
            messages.append(message)
    elif args.command == "smoke":
        emit(invoke(remote(args.checkpoint).smoke, [args.run_id], args.detach))
    elif args.command in ("intervene", "graph", "attribute", "fidelity"):
        emit(
            invoke(
                getattr(remote(args.checkpoint), args.command),
                [json_file(args.request)],
                args.detach,
            )
        )
    elif args.command == "dataset":
        from .datasets import build_dataset

        emit(build_dataset(json_file(args.views), args.output))
    elif args.command == "train":
        if args.remote:
            import modal

            method = modal.Function.from_name(
                app_name(args.checkpoint), "train_dictionary", environment_name="FDR"
            )
            emit(invoke(method, [json_file(args.config)], args.detach))
        else:
            from .dictionary import train_from_config

            emit(train_from_config(json_file(args.config)))
    elif args.command == "localize":
        from .analysis import localize

        emit(localize(args.run, json_file(args.pairs), args.sites, args.output))
    elif args.command == "compare-routes":
        from .analysis import router_comparison

        emit(router_comparison(args.run_a, args.run_b, args.layers, args.output))
    elif args.command == "features":
        from .analysis import feature_report

        emit(feature_report(args.dictionary, args.dataset, args.output))
    elif args.command == "status":
        apps = json.loads(
            subprocess.check_output(
                [sys.executable, "-m", "modal", "app", "list", "--env=FDR", "--json"], text=True
            )
        )
        emit([a for a in apps if a["description"].startswith("k2-interp-")])


if __name__ == "__main__":
    main()
