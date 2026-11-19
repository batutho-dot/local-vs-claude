# Report

**What it does:** prints the results table (three scores, time, memory) for the Claude
reference and every file in `runs/`, plus the misses per model.

**How to reach it:** `python run.py`.

**How verify drives it:** `scores` step scores every saved run with rag-eval's scorer and
compares found / right / declined / wrong ids with `verify_known_good.json`, then runs
`report.main()` and checks three local rows are printed.

**End state:** the same numbers as the README table.

**Gotchas:** a new run file for a model not in `verify_known_good.json` fails the step on
purpose; add it there in the same commit.
