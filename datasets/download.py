"""Download the frozen public ITRDB RWL list with atomic SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(row: dict, root: Path) -> dict:
    destination = root / row["relativePath"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == row["sha256"]:
        return {"id": row["id"], "status": "cached", "path": str(destination)}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = Request(row["url"], headers={"User-Agent": "Crossdating-Model/5.0.0"})
    digest = hashlib.sha256()
    with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    actual = digest.hexdigest()
    if actual != row["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {row['relativePath']}: {actual}")
    temporary.replace(destination)
    return {"id": row["id"], "status": "downloaded", "path": str(destination), "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append", help="Relative RWL id/path; repeatable")
    parser.add_argument("--role", choices=("development", "calibration"))
    parser.add_argument("--root", type=Path, default=HERE / "rwl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((HERE / "source-manifest.json").read_text(encoding="utf-8"))
    rows = manifest["sources"]
    if args.role:
        rows = [row for row in rows if row["role"] == args.role]
    if args.asset:
        requested = set(args.asset)
        rows = [row for row in rows if row["id"] in requested or row["relativePath"] in requested]
        found = {row["id"] for row in rows} | {row["relativePath"] for row in rows}
        unknown = [item for item in requested if item not in found]
        if unknown:
            raise SystemExit(f"unknown assets: {unknown}")
    if not rows:
        raise SystemExit("no RWL files selected")
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row, args.root) for row in rows]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            counts[result["status"]] += 1
            if number % 25 == 0 or number == len(rows):
                print(json.dumps({"verified": number, "total": len(rows), **counts}), flush=True)
    print(json.dumps({"root": str(args.root.resolve()), "verified": len(rows), **counts}, indent=2))


if __name__ == "__main__":
    main()
