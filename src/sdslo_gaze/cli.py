"""Command-line interface: ``sdslo-gaze <command>``."""

from __future__ import annotations

import argparse
import json

from . import pipeline
from .scan_timing import TIMING_MODELS


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--which", default="test1", help="capture id (default: test1)")
    p.add_argument("--timing", default="flyback10ms", choices=sorted(TIMING_MODELS),
                   help="active/flyback timing assumption")
    p.add_argument("--target-hz", type=float, default=960.0, help="target strip rate (default 960)")
    p.add_argument("--source", default="reference", choices=["reference", "frames"],
                   help="'reference' uses cached strip track; 'frames' runs the tracker")
    p.add_argument("--S", type=int, default=None, help="strip width override")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--data-root", default=None, help="path to the source data tree")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--no-dot", action="store_true", help="skip dot correlation")


def _run(args) -> int:
    res = pipeline.run(
        which=args.which, timing_model=args.timing, target_hz=args.target_hz,
        source=args.source, S=args.S, max_frames=args.max_frames,
        data_root=args.data_root, out_dir=args.out_dir, with_dot=not args.no_dot,
    )
    summary = {
        "which": res.which,
        "strip_hz": res.strip_hz,
        "strip_width": res.strip_width,
        "n_observed": res.n_observed,
        "role_counts": res.role_counts,
        "precision_arcmin": res.precision_arcmin,
        "microsaccades": res.microsaccades,
        "dot_correlation": res.dot_correlation,
        "manifest": res.manifest_path,
    }
    print(json.dumps(pipeline._to_jsonable(summary), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sdslo-gaze", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_track = sub.add_parser("track", help="run the full pipeline and write a manifest")
    _add_common(p_track)
    p_track.set_defaults(func=_run)

    p_val = sub.add_parser("validate", help="alias for track --source reference (reproduce reference)")
    _add_common(p_val)
    p_val.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
