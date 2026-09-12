"""Command line: ``python -m pipeline <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _plan(args) -> int:
    from pipeline import eligibility, families, hub, settings, sizing

    cfg = settings.load()
    source = hub.fetch(args.model, args.revision)
    verdict = eligibility.evaluate(source, cfg)
    result = verdict.to_dict()
    family = families.family_for(source.config) if source.config else None
    if family and source.total_params:
        arch = families.architecture(source.config, source.total_params)
        choice = sizing.choose_context(
            arch, cfg.context_tiers, cfg.device_budget_bytes, cfg.runtime_overhead_bytes, None
        )
        result["window"] = {"context": choice.context, "reason": choice.reason, "table": list(choice.table)}
    print(json.dumps(result, indent=2, default=str))
    return 0 if verdict.eligible else 3


def _export_xnnpack(args) -> int:
    from pipeline import export_xnnpack

    try:
        report = export_xnnpack.run(
            args.model,
            args.revision,
            Path(args.out),
            Path(args.work),
            context=args.context,
            keep_work=args.keep_work,
            skip_smoke=args.skip_smoke,
        )
    except export_xnnpack.ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"file": report["files"], "window": report["window"]["context"]}, indent=2))
    return 0


def _publish_hf(args) -> int:
    from pipeline import publish

    print(publish.publish_hf(Path(args.out), args.backend, args.target))
    return 0


def _publish_release(args) -> int:
    from pipeline import publish

    print(publish.publish_release(Path(args.out), args.backend, args.target))
    return 0


def _summary(args) -> int:
    from pipeline import publish

    sys.stdout.write(publish.summary(Path(args.out), args.backend, args.target))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="eligibility and window choice for one model, no download")
    plan.add_argument("model")
    plan.add_argument("--revision", default="main")
    plan.set_defaults(func=_plan)

    export = commands.add_parser("export-xnnpack", help="download, convert, export and smoke-test")
    export.add_argument("model")
    export.add_argument("--revision", default="main")
    export.add_argument("--out", default="out")
    export.add_argument("--work", default="work")
    export.add_argument("--context", type=int, default=None, help="force a window instead of auto-fit")
    export.add_argument("--keep-work", action="store_true")
    export.add_argument("--skip-smoke", action="store_true")
    export.set_defaults(func=_export_xnnpack)

    for name, func, text in (
        ("publish-hf", _publish_hf, "commit one backend folder to the output HF repo"),
        ("publish-release", _publish_release, "attach one backend's files to a GitHub release"),
        ("summary", _summary, "markdown summary of one backend's export"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("out")
        command.add_argument("--backend", required=True, choices=["xnnpack", "qnn", "mtk"])
        command.add_argument("--target", default=None)
        command.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
