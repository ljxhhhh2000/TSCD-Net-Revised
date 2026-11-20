from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import evaluate_user_symptom_alignment
from .analysis import analyze_symptom_trajectories
from .config import Config, PROJECT_ROOT
from .experiment import prepare, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tscdnet",
        description="Reproduce TSCD-Net on SWDD.",
    )
    parser.add_argument(
        "command",
        choices=("prepare", "train", "all", "analyze", "align"),
        help=(
            "Prepare features, train, run both, export trajectories, or "
            "evaluate symptom alignment."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/chronological.json",
        help="Path to the JSON experiment configuration.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--symptom-keys")
    parser.add_argument("--output-dir")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--models")
    parser.add_argument(
        "--max-users-per-class",
        type=int,
        help="Limit smoke runs; omit for the full SWDD population.",
    )
    parser.add_argument(
        "--scan-limit-per-file",
        type=int,
        help="Limit JSONL lines read from each source file.",
    )
    parser.add_argument(
        "--include-retweets",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config.from_json(args.config)
    _apply_overrides(config, args)
    if args.command in {"prepare", "all"}:
        prepare(config, force=args.force)
    if args.command in {"train", "all"}:
        output = train(config)
        print(f"Wrote metrics summary: {output}")
    if args.command == "analyze":
        output = analyze_symptom_trajectories(config)
        print(f"Wrote symptom analysis: {output}")
    if args.command == "align":
        output = evaluate_user_symptom_alignment(config)
        print(f"Wrote symptom alignment report: {output}")


def _apply_overrides(
    config: Config,
    args: argparse.Namespace,
) -> None:
    if args.dataset_path is not None:
        config.data.dataset_zip = args.dataset_path
    if args.sequence_length is not None:
        config.features.sequence_length = args.sequence_length
    if args.symptom_keys is not None:
        config.experiment.symptom_keys = [
            value.strip()
            for value in args.symptom_keys.split(",")
            if value.strip()
        ]
    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir
    if args.artifact_dir is not None:
        config.experiment.artifact_dir = args.artifact_dir
    if args.runs is not None:
        config.training.runs = args.runs
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.model.device = args.device
    if args.models is not None:
        config.experiment.models = [
            value.strip()
            for value in args.models.split(",")
            if value.strip()
        ]
    if args.max_users_per_class is not None:
        config.data.max_users_per_class = args.max_users_per_class
    if args.scan_limit_per_file is not None:
        config.data.scan_limit_per_file = args.scan_limit_per_file
    if args.include_retweets is not None:
        config.data.include_retweets = args.include_retweets
    config.validate()


if __name__ == "__main__":
    main()
