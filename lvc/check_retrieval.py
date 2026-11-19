"""Check that the saved passages are what the hybrid search returns today.

Reruns rag-eval's hybrid search (TF-IDF + bge-small, reciprocal rank fusion) for all 30
questions and compares the top-5 chunk ids with reference/haiku_hybrid.json. Needs fastembed
(pip install fastembed==0.9.1); downloads the small embedding model on first use.
"""

import json
import sys

from lvc.generate import REFERENCE
from pipeline.chunk import load_chunks
from pipeline.hybrid import HybridIndex
from rageval.config import DATA_TEXT, K, TESTSET


def main() -> None:
    ref = {r["id"]: r["retrieved"] for r in json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]}
    chunks = load_chunks(DATA_TEXT)
    index = HybridIndex([c.text for c in chunks])
    diff = []
    for q in json.loads(TESTSET.read_text(encoding="utf-8")):
        got = [chunks[i].id for i, _ in index.search(q["question"], K)]
        if got != ref[q["id"]]:
            diff.append(q["id"])
    print(f"retrieval matches the saved Claude run on {len(ref) - len(diff)}/{len(ref)} questions"
          + (f"; differs on {diff}" if diff else ""))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
