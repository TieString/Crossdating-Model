from __future__ import annotations
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from crossdating_model.config import file_sha256, load_config, path_sha256
from crossdating_model.evaluation.workflow import evaluate, gate_for_clean, grouped_summaries, summarize
from crossdating_model.evaluation.bootstrap import file_clustered_interval

def profile_operations(scores: np.ndarray, identities: np.ndarray, starts: np.ndarray,
                       identity_count: int) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(identity_count, -1e9)
    windows = np.zeros(identity_count, dtype=np.int32)
    for identity in np.unique(identities):
        members = np.where(identities == identity)[0]
        winner = int(members[np.argmax(scores[members])])
        result[identity] = scores[winner]
        windows[identity] = starts[winner]
    return result, windows

def load_packed(dataset_path: str | Path, manifest_path: str | Path) -> tuple[dict, dict]:
    dataset_path, manifest_path = Path(dataset_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_path.is_dir():
        packed = {key: np.load(dataset_path / f"{key}.npy", mmap_mode="r")
                  for key in ("x", "y", "offsets", "sizes", "codes", "starts")}
        expected = (dataset_path / "manifest-sha256.txt").read_text(encoding="ascii").strip()
        if expected != file_sha256(manifest_path):
            raise ValueError("packed directory and manifest SHA-256 disagree")
    elif dataset_path.suffix == ".joblib":
        packed = joblib.load(dataset_path, mmap_mode="r")
        expected = packed.get("manifestSha256")
        if expected and expected != file_sha256(manifest_path):
            raise ValueError("packed dataset and manifest SHA-256 disagree")
    else:
        source = np.load(dataset_path, allow_pickle=False)
        packed = {key: source[key] for key in source.files}
    required = {"x", "y", "offsets", "sizes", "codes", "starts"}
    if required - packed.keys():
        raise ValueError(f"missing packed arrays: {sorted(required - packed.keys())}")
    if len(manifest["rows"]) != len(packed["sizes"]):
        raise ValueError("row/group count mismatch")
    return packed, manifest

def train(dataset_path: str | Path, manifest_path: str | Path, config_path: str | Path,
          output_dir: str | Path, folds: int | None = None) -> dict:
    config = load_config(config_path)
    packed, manifest = load_packed(dataset_path, manifest_path)
    rows, identities = manifest["rows"], manifest["identities"]
    x, y = packed["x"], packed["y"]
    offsets, sizes, codes, starts = (packed[key] for key in ("offsets", "sizes", "codes", "starts"))
    if (x.shape[1] != config["feature_count"]
            or len(manifest["columns"]) != config["feature_count"]
            or len(set(manifest["columns"])) != config["feature_count"]):
        raise ValueError("feature schema mismatch")
    if not np.array_equal(offsets[1:] - offsets[:-1], sizes):
        raise ValueError("candidate offsets and group sizes disagree")
    development = np.asarray([i for i, row in enumerate(rows) if row["role"] == "development"])
    calibration = np.asarray([i for i, row in enumerate(rows) if row["role"] == "calibration"])
    if not len(development) or not len(calibration):
        raise ValueError("both development and calibration states are required")
    dev_groups = np.asarray([rows[i].get("evaluationClusterId", rows[i]["fileContentHash"]) for i in development])
    cal_groups = {rows[i].get("evaluationClusterId", rows[i]["fileContentHash"]) for i in calibration}
    if set(dev_groups) & cal_groups:
        raise ValueError("development/calibration file-group leakage")
    def flattened(indices: np.ndarray) -> np.ndarray:
        return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
    parameters = dict(config["lightgbm"])
    jobs = parameters.pop("n_jobs")
    parameters.update(objective="lambdarank", label_gain=[0, 1], n_jobs=jobs,
                      verbosity=-1, random_state=config["seed"])
    def fit(indices: np.ndarray, seed: int):
        selected = flattened(indices)
        model = lgb.LGBMRanker(**{**parameters, "random_state": seed})
        model.fit(x[selected], y[selected], group=[int(sizes[i]) for i in indices])
        return model
    def predict(indices: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
        score_table = np.full((len(indices), len(identities)), -1e9)
        window_table = np.zeros_like(score_table, dtype=np.int32)
        for position, index in enumerate(indices):
            selection = slice(offsets[index], offsets[index + 1])
            candidate_scores = model.booster_.predict(x[selection], num_threads=1)
            score_table[position], window_table[position] = profile_operations(
                candidate_scores, codes[selection], starts[selection], len(identities))
        return score_table, window_table
    fold_count = folds or int(config["evaluation"]["file_grouped_folds"])
    if len(set(dev_groups)) < fold_count:
        raise ValueError("not enough development file groups for cross-validation")
    oof_scores = np.full((len(development), len(identities)), -1e9)
    oof_windows = np.zeros_like(oof_scores, dtype=np.int32)
    for fold, (training, testing) in enumerate(GroupKFold(fold_count).split(development, groups=dev_groups), 1):
        current = fit(development[training], config["seed"] + fold)
        oof_scores[testing], oof_windows[testing] = predict(development[testing], current)
    none = identities.index(["none", 0])
    development_rows = [rows[i] for i in development]
    development_gate = gate_for_clean(development_rows, oof_scores, none,
                                      config["evaluation"]["clean_false_positive_max"])
    development_result = evaluate(development_rows, oof_scores, oof_windows, identities, development_gate)
    model = fit(development, config["seed"])
    calibration_scores, calibration_windows = predict(calibration, model)
    calibration_rows = [rows[i] for i in calibration]
    gate = gate_for_clean(calibration_rows, calibration_scores, none,
                          config["evaluation"]["clean_false_positive_max"])
    calibration_result = evaluate(calibration_rows, calibration_scores, calibration_windows, identities, gate)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {"version": config["asset_model_version"], "releaseVersion": config["model_version"],
        "model": model, "columns": manifest["columns"], "proposalSpec": manifest.get("spec", {}),
        "eventGate": gate, "identities": identities, "windowWidth": config["window_width"],
        "operationThenConditionalWindow": True, "config": config}
    model_path = output / "model.joblib"
    joblib.dump(artifact, model_path)
    report = {"schemaVersion": 1, "modelVersion": config["model_version"], "datasetSha256": path_sha256(dataset_path),
        "manifestSha256": file_sha256(manifest_path), "configSha256": file_sha256(config_path),
        "developmentFiles": len(set(dev_groups)), "calibrationFiles": len(cal_groups), "fileOverlap": 0,
        "developmentOof": summarize(development_result), "calibration": summarize(calibration_result),
        "calibrationGroups": grouped_summaries(calibration_result),
        "calibrationBootstrap": file_clustered_interval(calibration_result,
            int(config["evaluation"]["bootstrap_iterations"]), config["seed"]),
        "parameters": config["lightgbm"], "newFinalConsumed": False}
    (output / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "scores.npz", developmentScores=oof_scores,
                       developmentWindows=oof_windows, calibrationScores=calibration_scores,
                       calibrationWindows=calibration_windows)
    return {"artifact": str(model_path), "report": report}
