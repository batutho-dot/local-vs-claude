"""Print the results table from the saved runs. Spends nothing and calls no model.

Scores use rag-eval's own rule-based scorer (rageval/report.py, pinned copy):
  1. right passage found   (same for every row: retrieval is fixed)
  2. right answer given    (all required answer terms present, not declined)
  3. unanswerable declined (answered NOT IN SOURCES)
"""

import json
import statistics

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


def main() -> None:
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    chunk_text = {c.id: c.text for c in load_chunks(DATA_TEXT)}
    runs = [json.loads(REFERENCE.read_text(encoding="utf-8"))]
    runs += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))]

    print("| Model | Where | Size, quantization | Right passage found | Right answer given | "
          "Unanswerable declined | Median s / answer | Total s (30) | Memory used |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        s = summarise(run, questions, chunk_text)
        if "memory_bytes" in run:
            where = "local"
            size = f"{run['parameter_size']}, {run['quantization']}, {run['file_bytes'] / 1e9:.1f} GB file"
            mem = f"{run['memory_bytes'] / 1e9:.1f} GB ({run['memory_vram_bytes'] / 1e9:.1f} GB on GPU)"
        else:
            where, size, mem = "Anthropic API", "not published", "n/a (cloud)"
        print(f"| {run['model']} | {where} | {size} | {s['found']}/{s['n_ans']} | {s['right']}/{s['n_ans']} | "
              f"{s['declined']}/{s['n_una']} | {s['median_s']:.1f} | {s['total_s']:.0f} | {mem} |")

    print("\nMisses per model:")
    for run in runs:
        s = summarise(run, questions, chunk_text)
        print(f"  {run['model']}: wrong {s['wrong'] or 'none'}; falsely declined {s['false_decline'] or 'none'}; "
              f"answered an unanswerable {s['answered_unanswerable'] or 'none'}")
