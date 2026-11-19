"""Answer the 30 questions with Claude Haiku 5.5 on the same fixed passages, for repeat runs.

Same model, settings and prompt as rag-eval's generate step (claude-haiku-5-5, effort low,
max_tokens 4000), so a new run here is a repeat of the saved reference run. Writes
runs/claude-haiku-5-5_<tag>.json with answers, seconds, tokens and cost. Stops before the
total Haiku spend in runs/ passes MAX_TOTAL_USD.

The key is read from ANTHROPIC_API_KEY; on Windows, if the variable is not in this process,
the user-level variable is read from the registry. The key is never printed or written.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from lvc.generate import REFERENCE, RUNS, SYSTEM, prompt
from pipeline.chunk import load_chunks
from rageval.config import DATA_TEXT, GENERATOR, TESTSET, cost_usd

MAX_TOTAL_USD = 0.50


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or sys.platform != "win32":
        return
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
    except OSError:
        pass


def spent_so_far() -> float:
    return sum(json.loads(p.read_text(encoding="utf-8")).get("cost_usd", 0.0)
               for p in RUNS.glob(f"{GENERATOR}_*.json"))


def main(tag: str) -> None:
    import anthropic

    load_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; no API call made.")
    out = RUNS / f"{GENERATOR}_{tag}.json"
    if out.exists():
        sys.exit(f"{out.name} exists; pick another tag")
    budget = MAX_TOTAL_USD - spent_so_far()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    chunks = {c.id: c.text for c in load_chunks(DATA_TEXT)}
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    client = anthropic.Anthropic()

    rows = []
    for ref_row in ref["rows"]:
        q = questions[ref_row["id"]]
        passages = [chunks[c] for c in ref_row["retrieved"]]
        t = time.perf_counter()
        msg = client.messages.create(
            model=GENERATOR, max_tokens=4000, system=SYSTEM, output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt(q["question"], passages)}],
        )
        u = msg.usage
        rows.append({
            "id": q["id"], "retrieved": ref_row["retrieved"],
            "answer": "".join(b.text for b in msg.content if b.type == "text").strip(),
            "stop_reason": msg.stop_reason, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cost_usd": cost_usd(GENERATOR, u.input_tokens, u.output_tokens),
            "seconds": round(time.perf_counter() - t, 2),
        })
        if sum(r["cost_usd"] for r in rows) > budget:
            sys.exit(f"Haiku spend would pass {MAX_TOTAL_USD} USD in total; stopped, nothing saved")
        print(f"  {q['id']}  {rows[-1]['seconds']:>5.1f} s  {rows[-1]['answer'][:80]!r}")

    run = {
        "model": GENERATOR, "tag": tag, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6), "rows": rows,
    }
    RUNS.mkdir(exist_ok=True)
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{GENERATOR} {tag}: 30 answers, {run['cost_usd']:.4f} USD -> {out.name}")
