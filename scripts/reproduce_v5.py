"""Run the frozen v5 training/export/parity pipeline.

`--smoke` uses a compact fixture derived from the real v5 candidate matrix and
proves pipeline mechanics. Full scientific reproduction requires the separately
published frozen evidence bundle declared in datasets/source-manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy
import scipy
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdating_model.config import file_sha256, load_config
from crossdating_model.export.gold import generate_gold
from crossdating_model.export.lightgbm_json import export_model
from crossdating_model.training.ranker import train


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v5.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-typescript", action="store_true")
    args = parser.parse_args()
    if args.smoke and (args.dataset or args.manifest):
        parser.error("--smoke cannot be combined with explicit dataset paths")
    dataset = args.dataset or ROOT / ("datasets/fixtures/v5-smoke.npz" if args.smoke
                                      else "datasets/cache/packed-data.joblib")
    manifest = args.manifest or ROOT / ("datasets/fixtures/v5-smoke-manifest.json" if args.smoke
                                        else "datasets/cache/full-manifest.json")
    if not dataset.exists() or not manifest.exists():
        parser.error(f"missing frozen inputs: {dataset} / {manifest}")
    output = args.output or ROOT / "artifacts" / ("smoke" if args.smoke else "v5-full")
    started = datetime.now(timezone.utc)
    config = load_config(args.config)
    trained = train(dataset, manifest, args.config, output)
    model_path = output / "unifiedV5Model.json"
    export_model(trained["artifact"], model_path)
    smoke_config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text(encoding="utf-8"))
    gold_path = output / "python-gold.json"
    generate_gold(trained["artifact"], dataset, manifest, gold_path,
                  limit=int(smoke_config["gold_states"]) if args.smoke else 400)
    finished = datetime.now(timezone.utc)
    run = {
        "schemaVersion": 1,
        "mode": "smoke" if args.smoke else "full",
        "scientificAccuracyClaim": not args.smoke,
        "gitCommit": git_commit(),
        "seed": config["seed"],
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikitLearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "lightgbm": lightgbm.__version__,
        "cofechaJs": config["cofecha_js_version"],
        "configSha256": file_sha256(args.config),
        "datasetSha256": file_sha256(dataset),
        "inputManifestSha256": file_sha256(manifest),
        "modelSha256": file_sha256(model_path),
        "startedAt": started.isoformat(),
        "finishedAt": finished.isoformat(),
    }
    (output / "run-manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if not args.skip_typescript:
        environment = os.environ.copy()
        environment["CROSSDATING_MODEL_JSON"] = str(model_path)
        environment["CROSSDATING_GOLD_JSON"] = str(gold_path)
        npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if npm is None:
            raise RuntimeError("npm was not found on PATH")
        subprocess.run([npm, "run", "test:parity"], cwd=ROOT, env=environment, check=True)
    print(json.dumps({"output": str(output), "run": run,
                      "metrics": trained["report"]}, indent=2))


if __name__ == "__main__":
    main()
