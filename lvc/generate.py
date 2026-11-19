"""Answer every test question with a local model through Ollama, on the same passages Claude saw.

Retrieval is fixed: each question gets exactly the 5 chunks that the hybrid search retrieved
in rag-eval's Claude Haiku 5.5 run (reference/haiku_hybrid.json). So the only thing that
changes between rows of the results table is the model that writes the answer.

The system prompt and the user prompt are copied word for word from rag-eval's
rageval/generate.py, so the local models get the same instructions as Claude.

Per answer this records the wall-clock time and Ollama's own token counts. After the run it
records the memory the loaded model takes, as Ollama reports it in /api/ps (total, and the
part held in GPU memory). Talks only to the local Ollama server on 127.0.0.1.
"""

import json
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pipeline.chunk import load_chunks
from rageval.config import DATA_TEXT, DECLINE, TESTSET

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference" / "haiku_hybrid.json"
RUNS = ROOT / "runs"
OLLAMA = "http://127.0.0.1:11434"

# Settings held the same for every local model.
NUM_CTX = 8192        # context window in tokens; the longest prompt here is well under it
TEMPERATURE = 0.0     # always pick the most likely next word, so a rerun gives the same answer
NUM_PREDICT = 1024    # cap on answer length in tokens

SYSTEM = (  # verbatim from rag-eval rageval/generate.py
    "You answer questions about EU sustainability law using only the numbered passages in "
    "the user message. Do not use any other knowledge. If the passages do not contain the "
    f"answer, reply with exactly: {DECLINE}\n"
    "Otherwise answer in one to three sentences and name the passage numbers you used in "
    "square brackets, for example [2]."
)


def prompt(question: str, passages: list[str]) -> str:  # verbatim from rag-eval
    numbered = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return f"Passages:\n\n{numbered}\n\nQuestion: {question}"


def call(path: str, body: dict | None = None, timeout: int = 600) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_info(model: str) -> dict:
    show = call("/api/show", {"model": model})
    d = show.get("details", {})
    tags = {m["name"]: m for m in call("/api/tags")["models"]}
    return {
        "parameter_size": d.get("parameter_size"),
        "quantization": d.get("quantization_level"),
        "family": d.get("family"),
        "file_bytes": tags.get(model, {}).get("size"),
        "digest": tags.get(model, {}).get("digest"),
        "capabilities": show.get("capabilities", []),
    }


def chat(model: str, user: str, thinking: bool) -> dict:
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        "keep_alive": "10m",
    }
    if thinking:
        body["think"] = False  # answer directly, like Claude at low effort; no hidden reasoning
    t = time.perf_counter()
    r = call("/api/chat", body)
    return {
        "answer": r["message"]["content"].strip(),
        "seconds": round(time.perf_counter() - t, 2),
        "input_tokens": r.get("prompt_eval_count"),
        "output_tokens": r.get("eval_count"),
        "done_reason": r.get("done_reason"),
    }


RAM_MARGIN_BYTES = 3_000_000_000  # free RAM needed beyond the model's own memory before it may load


def free_ram_bytes() -> int | None:
    """Free physical memory, from Windows (GlobalMemoryStatusEx) or /proc/meminfo; None if unknown."""
    if platform.system() == "Windows":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def preflight(model: str, file_bytes: int | None) -> dict:
    """One model at a time: refuse to load if another model is loaded or RAM is short."""
    loaded = [m["name"] for m in call("/api/ps")["models"]]
    if loaded:
        raise SystemExit(f"another model is loaded ({loaded}); unload it first, nothing run")
    free = free_ram_bytes()
    need = model_bytes(model) + RAM_MARGIN_BYTES
    if free is not None and free < need:
        raise SystemExit(f"only {free / 1e9:.1f} GB RAM free, need {need / 1e9:.1f} GB for {model}; nothing run")
    return {"free_ram_before_bytes": free}


def model_bytes(model: str) -> int:
    """Memory the model needs: the largest a saved run measured, else its file size."""
    measured = [json.loads(p.read_text(encoding="utf-8")).get("memory_bytes") or 0
                for p in RUNS.glob(model.replace(":", "_").replace("/", "_") + "_*.json")]
    return max(measured + [model_info(model)["file_bytes"] or 0])


def wait_for_ram(model: str, minutes: int = 180) -> bool:
    """Wait until free RAM covers the model's memory plus the margin, rechecking every
    3 minutes. False after `minutes`."""
    need = model_bytes(model) + RAM_MARGIN_BYTES
    for _ in range(0, minutes, 3):
        free = free_ram_bytes()
        if free is None or free >= need:
            return True
        print(f"  waiting for RAM: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed for {model}", flush=True)
        time.sleep(180)
    return False


def unload(model: str) -> None:
    call("/api/generate", {"model": model, "keep_alive": 0})


def main(model: str, tag: str = "r1") -> None:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    chunks = {c.id: c.text for c in load_chunks(DATA_TEXT)}
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    info = model_info(model)
    thinking = "thinking" in info["capabilities"]
    version = call("/api/version")["version"]
    pre = preflight(model, info["file_bytes"])

    # Warm-up: load the model into memory first, so load time is reported apart from answer time.
    t = time.perf_counter()
    chat(model, "Passages:\n\n[1] Test.\n\nQuestion: Say OK.", thinking)
    load_s = round(time.perf_counter() - t, 2)

    rows = []
    for ref_row in ref["rows"]:
        q = questions[ref_row["id"]]
        passages = [chunks[c] for c in ref_row["retrieved"]]
        row = {"id": q["id"], "retrieved": ref_row["retrieved"]}
        row.update(chat(model, prompt(q["question"], passages), thinking))
        rows.append(row)
        print(f"  {q['id']}  {row['seconds']:>6.1f} s  {row['answer'][:80]!r}")

    loaded = [m for m in call("/api/ps")["models"] if m["name"] == model or m["model"] == model]
    mem = loaded[0] if loaded else {}
    unload(model)
    run = {
        "model": model,
        "tag": tag,
        **info,
        **pre,
        "ollama_version": version,
        "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "num_predict": NUM_PREDICT,
                    "think": False if thinking else None},
        "machine": {"os": platform.platform(), "cpu": platform.processor()},
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warmup_seconds": load_s,
        "memory_bytes": mem.get("size"),
        "memory_vram_bytes": mem.get("size_vram"),
        "total_seconds": round(sum(r["seconds"] for r in rows), 2),
        "rows": rows,
    }
    RUNS.mkdir(exist_ok=True)
    out = RUNS / (model.replace(":", "_").replace("/", "_") + f"_{tag}.json")
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{model}: {len(rows)} answers in {run['total_seconds']} s, memory "
          f"{(run['memory_bytes'] or 0) / 1e9:.2f} GB ({(run['memory_vram_bytes'] or 0) / 1e9:.2f} GB on GPU) -> {out.name}")
