from __future__ import annotations
import hashlib
from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if value["feature_count"] != 207 or value["window_width"] != 13:
        raise ValueError("v5 contract requires 207 features and a 13-year window")
    if value["cofecha_js_version"] != "0.2.0":
        raise ValueError("v5 evidence is frozen to cofecha-js 0.2.0")
    return value

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def path_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    digest = hashlib.sha256()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(child.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
