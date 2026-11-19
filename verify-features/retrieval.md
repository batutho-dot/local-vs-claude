# Retrieval

**What it does:** gives every model the same passages: the chunk ids rag-eval's hybrid
search retrieved for Claude, stored in `reference/haiku_hybrid.json`.

**How to reach it:** used inside `generate`; `python -m lvc.check_retrieval` reruns the search.

**How verify drives it:** `scores` checks the corpus splits into 1,257 chunks, so the saved
ids still point to the same text. `--full` runs `check_retrieval` (30/30 must match).

**End state:** 30/30 questions retrieve the saved chunk ids.

**Gotchas:** `check_retrieval` needs fastembed and downloads the embedding model the first time.
