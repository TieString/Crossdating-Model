"""Build the immutable public ITRDB download list from frozen file splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdating_model.config import file_sha256

BASE_URL = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements"
REGION_PREFIXES = {
    "australia": ("ausl", "newz"),
    "africa": ("dza", "mar", "morc"),
    "asia": ("chin", "chn", "jord", "kaz", "lbn", "mng", "mong", "pak", "paki", "rus", "russ"),
    "europe": ("aust", "aut", "bel", "blr", "brit", "che", "cyp", "czec", "deu", "finl",
               "fra", "fran", "germ", "grc", "gree", "lith", "norw", "pola", "spai", "svk",
               "swed", "swit", "tur", "turk"),
}


def remote_path(relative_path: str) -> str:
    if "/" in relative_path:
        return relative_path
    lowered = relative_path.lower()
    for region, prefixes in REGION_PREFIXES.items():
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return f"{region}/{relative_path}"
    raise ValueError(f"no ITRDB region mapping for {relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "splits/FILE_SPLITS.json")
    parser.add_argument("--local-root", type=Path,
                        help="Optional existing ITRDB measurements tree; verifies bytes and records sizes")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/source-manifest.json")
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    sources = []
    seen: set[str] = set()
    for role, rows in sorted(splits["roles"].items()):
        for row in rows:
            relative = row["relativePath"]
            if relative in seen:
                raise ValueError(f"RWL appears in multiple roles: {relative}")
            seen.add(relative)
            source = args.local_root / relative if args.local_root else None
            if source is not None:
                if not source.is_file():
                    raise FileNotFoundError(source)
                actual = file_sha256(source)
                if actual != row["sha256"]:
                    raise ValueError(f"local RWL SHA-256 mismatch for {relative}: {actual}")
            remote = remote_path(relative)
            sources.append({
                "id": relative.removesuffix(".rwl").replace("/", "-"),
                "kind": "rwl",
                "role": role,
                "relativePath": relative,
                "url": f"{BASE_URL}/{remote}",
                "sha256": row["sha256"],
                **({"bytes": source.stat().st_size} if source is not None else {}),
            })
    payload = {
        "schemaVersion": 2,
        "provider": "NOAA/NCEI International Tree-Ring Data Bank",
        "baseUrl": BASE_URL,
        "generatedFrom": "splits/FILE_SPLITS.json",
        "sourceCount": len(sources),
        "derivedEvidencePublished": False,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "sources": len(sources),
                      "sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
