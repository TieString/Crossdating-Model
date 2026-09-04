"""Require a reconstructed model to match the accepted language-neutral model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdating_model.config import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "model-releases/v5.0.0/unifiedV5Model.json")
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    fields = ("version", "columns", "proposalSpec", "eventGate", "identities",
              "windowWidth", "operationThenConditionalWindow", "objective",
              "averageOutput", "treeInfo")
    mismatches = [field for field in fields if candidate.get(field) != reference.get(field)]
    if mismatches:
        raise ValueError(f"reconstructed model differs in: {mismatches}")
    print(json.dumps({
        "result": "PASS",
        "semanticFieldsExact": list(fields),
        "candidateSha256": file_sha256(args.candidate),
        "referenceSha256": file_sha256(args.reference),
        "referenceAssetSha256": "4a355af59c22a9cd53e20d233256d94b44b83aea222d7ace55e6cb7e9bbb5c1e",
    }, indent=2))


if __name__ == "__main__":
    main()
