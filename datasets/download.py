"""Download only declared public assets and verify them before atomic commit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append", help="Declared source id; repeatable. Defaults to public RWL files.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
    requested = set(args.asset or [row["id"] for row in manifest["sources"] if row["kind"] == "rwl"])
    available = {row["id"]: row for row in manifest["sources"]}
    unknown = requested - available.keys()
    if unknown:
        raise SystemExit(f"Unknown asset ids: {sorted(unknown)}")
    for asset_id in sorted(requested):
        row = available[asset_id]
        if not row.get("url"):
            raise SystemExit(f"{asset_id} has no public URL yet; obtain it from the release/DOI listed in the manifest")
        destination = ROOT / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) == row["sha256"]:
            print(json.dumps({"id": asset_id, "status": "cached", "path": str(destination)}))
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        request = Request(row["url"], headers={"User-Agent": "Crossdating-Model/5.0.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        actual = sha256(temporary)
        if actual != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"SHA-256 mismatch for {asset_id}: {actual}")
        temporary.replace(destination)
        print(json.dumps({"id": asset_id, "status": "downloaded", "path": str(destination), "sha256": actual}))

if __name__ == "__main__":
    main()
