"""Rebuild every derived v5 artifact from the public ITRDB RWL list."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "crossdating_model/evidence/reference_v5"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwl-root", type=Path, default=ROOT / "datasets/rwl")
    parser.add_argument("--work", type=Path, default=ROOT / "work/v5-rebuild")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--evidence-limit", type=int, default=0,
                        help="Pipeline debugging only; zero means the complete frozen design")
    parser.add_argument("--skip-training", action="store_true",
                        help="Stop after rebuilding and packing all candidate evidence")
    args = parser.parse_args()
    if args.work.exists():
        raise FileExistsError(f"choose a new work directory: {args.work}")
    if args.download:
        run([sys.executable, "datasets/download.py", "--root", str(args.rwl_root),
             "--workers", str(args.workers)])
    args.work.mkdir(parents=True)
    tsx = shutil.which("tsx.cmd" if sys.platform == "win32" else "tsx")
    if tsx is None:
        local = ROOT / "node_modules/.bin" / ("tsx.cmd" if sys.platform == "win32" else "tsx")
        tsx = str(local) if local.exists() else None
    if tsx is None:
        raise RuntimeError("tsx is unavailable; run npm ci")
    indexes = []
    for role in ("development", "calibration"):
        index = args.work / "index" / f"{role}.jsonl"
        command = [tsx, "typescript/pipeline/generateEvidence.ts", "--role", role,
                   "--rwl-root", str(args.rwl_root), "--output", str(index),
                   "--evidence-root", str(args.work / "evidence")]
        if args.evidence_limit:
            command += ["--limit", str(args.evidence_limit)]
        run(command)
        indexes += ["--index", str(index)]
    base = args.work / "operation-core.npz"
    run([sys.executable, str(REFERENCE / "build-engine-operation-candidates.py"),
         *indexes, "--workers", str(args.workers), "--output", str(base),
         "--cache-root", str(args.work / "cache/operation")])
    augmented = args.work / "operation-constant-lag.npz"
    run([sys.executable, str(REFERENCE / "augment-constant-lag-evidence.py"),
         "--input", str(base), "--output", str(augmented),
         "--workers", str(args.workers)])
    joint_v1 = args.work / "joint-v1.json"
    run([sys.executable, str(REFERENCE / "build-joint-window-evidence.py"),
         *indexes, "--base-features", str(augmented), "--output", str(joint_v1),
         "--cache", str(args.work / "cache/joint-v1"), "--workers", str(args.workers)])
    joint_v2 = args.work / "joint-v2.json"
    run([sys.executable, str(REFERENCE / "append-joint-profile-evidence.py"),
         "--manifest", str(joint_v1), "--output", str(joint_v2),
         "--cache", str(args.work / "cache/joint-v2"), "--workers", str(args.workers)])
    global_v1 = args.work / "global-v1.json"
    run([sys.executable, str(REFERENCE / "build-global-baseline-evidence.py"),
         "--manifest", str(joint_v2), "--output", str(global_v1),
         "--cache", str(args.work / "cache/global"), "--workers", str(args.workers)])
    global_manifest = args.work / "global-v2.json"
    run([sys.executable, str(REFERENCE / "append-global-baseline-geometry.py"),
         "--manifest", str(global_v1), "--output", str(global_manifest),
         "--cache", str(args.work / "cache/global-v2"), "--workers", str(args.workers)])
    final_manifest = args.work / "full-manifest.json"
    run([sys.executable, str(REFERENCE / "combine-joint-global-evidence.py"),
         "--base-manifest", str(joint_v2), "--baseline-manifest", str(global_manifest),
         "--output", str(final_manifest), "--cache", str(args.work / "cache/final")])
    packed = args.work / "packed"
    run([sys.executable, "scripts/pack_candidates.py", "--manifest", str(final_manifest),
         "--output", str(packed)])
    if args.skip_training:
        return
    run([sys.executable, "scripts/reproduce_v5.py", "--dataset", str(packed),
         "--manifest", str(final_manifest), "--output", str(args.work / "trained")])
    run([sys.executable, "scripts/compare_model_release.py",
         "--candidate", str(args.work / "trained/unifiedV5Model.json")])


if __name__ == "__main__":
    main()
