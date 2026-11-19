"""Print the results table from the saved runs. Spends nothing and calls no model.

Scores use rag-eval's own rule-based scorer (rageval/report.py, pinned copy):
  1. right passage found   (same for every row: retrieval is fixed)
  2. right answer given    (all required answer terms present, not declined)
  3. unanswerable declined (answered NOT IN SOURCES)

Each model has several runs: Claude Haiku 5.5's first run is rag-eval's saved reference run,
the rest are files in runs/. The table shows the mean over runs and the min to max range.
"""

import json
import statistics
from collections import defaultdict

from lvc.generate import REFERENCE, RUNS
from pipeline.chunk import load_chunks
from rageval.config import DATA_TEXT, TESTSET
from rageval.report import declined, script_grades


def summarise(run: dict, questions: dict, chunk_text: dict) -> dict:
    g = script_grades(run, questions, chunk_text)
    ans = [i for i, q in questions.items() if q["type"] == "answerable"]
    una = [i for i, q in questions.items() if q["type"] == "unanswerable"]
    secs = [r["seconds"] for r in run["rows"]]
    return {
        "found": sum(g[i]["passage_found"] for i in ans), "n_ans": len(ans),
        "right": sum(g[i]["correct"] for i in ans),
        "declined": sum(g[i]["correct"] for i in una), "n_una": len(una),
        "false_decline": [r["id"] for r in run["rows"] if r["id"] in ans and declined(r["answer"])],
        "wrong": [i for i in ans if not g[i]["correct"]],
        "answered_unanswerable": [i for i in una if not g[i]["correct"]],
        "median_s": statistics.median(secs), "total_s": sum(secs),
    }


def load_runs() -> dict[str, list[dict]]:
    """model -> runs, in tag order. The saved reference run is Claude's r1."""
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    by_model = defaultdict(list)
    by_model[ref["model"]].append({**ref, "tag": "r1"})
    for p in sorted(RUNS.glob("*.json")):
        run = json.loads(p.read_text(encoding="utf-8"))
        by_model[run["model"]].append(run)
    return dict(by_model)


def spread(values: list[float], fmt: str = "{:.0f}") -> str:
    mean = statistics.mean(values)
    lo, hi = min(values), max(values)
    m, a, b = fmt.format(mean), fmt.format(lo), fmt.format(hi)
    return m if a == b else f"{m} ({a} to {b})"


def main() -> None:
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    chunk_text = {c.id: c.text for c in load_chunks(DATA_TEXT)}
    models = load_runs()

    print("| Model | Runs | Right passage found | Right answer given (of 20) | Unanswerable declined (of 10) | "
          "Median s per answer | Memory used |")
    print("|---|---|---|---|---|---|---|")
    for model, runs in models.items():
        s = [summarise(r, questions, chunk_text) for r in runs]
        if "memory_bytes" in runs[0]:
            mem = spread([r["memory_bytes"] / 1e9 for r in runs], "{:.1f}") + " GB"
        else:
            mem = "n/a (cloud)"
        print(f"| {model} | {len(runs)} | {spread([x['found'] for x in s])} / 20 | {spread([x['right'] for x in s], '{:.1f}')} | "
              f"{spread([x['declined'] for x in s], '{:.1f}')} | {spread([x['median_s'] for x in s], '{:.1f}')} | {mem} |")

    print("\nPer run:")
    for model, runs in models.items():
        for r in runs:
            x = summarise(r, questions, chunk_text)
            print(f"  {model} {r['tag']}: right {x['right']}/20, declined {x['declined']}/10, median {x['median_s']:.1f} s, "
                  f"wrong {x['wrong'] or 'none'}, falsely declined {x['false_decline'] or 'none'}, "
                  f"answered an unanswerable {x['answered_unanswerable'] or 'none'}")
