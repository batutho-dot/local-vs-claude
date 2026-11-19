"""Check this repository after a change. Prints one line per step and ends with
`verify: PASS` or `verify: FAIL at <step>: <reason>`. Changes nothing in the repo: run output
goes to a temporary folder. Never calls the Claude API and needs no key.

    python verify.py          fast checks: compile, scorer units, saved-run scores, the
                              generate step against a stand-in Ollama server (a few seconds)
    python verify.py --full   also reruns llama3.2:3b on all 30 questions through the real
                              local Ollama server (about 2 minutes, needs Ollama and the model)
                              and reruns the hybrid search (needs fastembed)

Needs the EU texts in data/text/ (python run.py download).
"""

import io
import json
import py_compile
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
KNOWN = json.loads((ROOT / "verify_known_good.json").read_text(encoding="utf-8"))


class Fail(Exception):
    pass


def check(cond, reason):
    if not cond:
        raise Fail(reason)


def load():
    from pipeline.chunk import load_chunks
    from rageval.config import DATA_TEXT, TESTSET

    check(DATA_TEXT.exists() and any(DATA_TEXT.glob("*.txt")), "no EU texts in data/text (run: python run.py download)")
    chunks = load_chunks(DATA_TEXT)
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    return chunks, questions


def step_compile(tmp):
    files = [f for d in ("", "lvc", "pipeline", "rageval") for f in sorted((ROOT / d).glob("*.py"))]
    for f in files:
        try:
            py_compile.compile(str(f), cfile=str(tmp / "pyc" / (f.name + "c")), doraise=True)
        except py_compile.PyCompileError as e:
            raise Fail(f"{f.name}: {e.msg.strip().splitlines()[-1]}")
    return f"{len(files)} files compile"


def step_units(tmp):
    from lvc.generate import SYSTEM, prompt
    from rageval.report import declined, terms_ok

    check(declined("NOT IN SOURCES") and not declined("E1-6 [2]"), "declined() wrong")
    q = {"answer_terms": [["capex", "capital expenditure"], ["turnover"]]}
    check(terms_ok(q, "Turnover and capital expenditure") and not terms_ok(q, "turnover only"), "terms_ok wrong")
    check("NOT IN SOURCES" in SYSTEM and SYSTEM.startswith("You answer questions about EU sustainability law"),
          "system prompt no longer matches rag-eval's")
    check(prompt("Q?", ["a", "b"]) == "Passages:\n\n[1] a\n\n[2] b\n\nQuestion: Q?", "prompt format changed")
    return "scorer and prompt units ok"


def step_scores(tmp):
    from lvc.generate import REFERENCE, RUNS
    from lvc.report import summarise

    chunks, questions = load()
    check(len(chunks) == KNOWN["chunks"], f"{len(chunks)} chunks, expected {KNOWN['chunks']}")
    chunk_text = {c.id: c.text for c in chunks}
    runs = [REFERENCE] + sorted(RUNS.glob("*.json"))
    seen = set()
    for p in runs:
        run = json.loads(p.read_text(encoding="utf-8"))
        exp = KNOWN["scores"].get(run["model"])
        check(exp is not None, f"{p.name}: model {run['model']} not in verify_known_good.json")
        s = summarise(run, questions, chunk_text)
        got = {"found": s["found"], "right": s["right"], "declined": s["declined"], "wrong": s["wrong"]}
        check(got == exp, f"{run['model']}: scores {got}, expected {exp}")
        seen.add(run["model"])
    check(seen == set(KNOWN["scores"]), f"saved runs {sorted(seen)} differ from known-good {sorted(KNOWN['scores'])}")
    buf = io.StringIO()
    with redirect_stdout(buf):
        from lvc import report

        report.main()
    check(buf.getvalue().count("| local |") == 3, "report table does not show 3 local rows")
    return f"{len(runs)} saved runs score as known-good; report prints"


class FakeOllama(BaseHTTPRequestHandler):
    """Stand-in for the Ollama API: declines every question, reports fixed memory."""

    seen_system = []

    def log_message(self, *a):
        pass

    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self.reply({"models": [{"name": "fake:1b", "size": 1_000_000_000, "digest": "abc"}]})
        elif self.path == "/api/version":
            self.reply({"version": "fake"})
        elif self.path == "/api/ps":
            self.reply({"models": [{"name": "fake:1b", "model": "fake:1b", "size": 2_000_000_000, "size_vram": 2_000_000_000}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self.reply({"details": {"parameter_size": "1B", "quantization_level": "Q4_K_M", "family": "fake"},
                        "capabilities": ["completion", "thinking"]})
        elif self.path == "/api/chat":
            FakeOllama.seen_system.append((body["messages"][0]["content"], body.get("think"), body["options"]))
            self.reply({"message": {"content": "NOT IN SOURCES"}, "prompt_eval_count": 10, "eval_count": 3, "done_reason": "stop"})


def step_generate_offline(tmp):
    from lvc import generate
    from lvc.report import summarise

    chunks, questions = load()
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    old = generate.OLLAMA, generate.RUNS
    generate.OLLAMA, generate.RUNS = f"http://127.0.0.1:{server.server_port}", tmp / "runs"
    try:
        with redirect_stdout(io.StringIO()):
            generate.main("fake:1b")
    finally:
        generate.OLLAMA, generate.RUNS = old
        server.shutdown()
    run = json.loads((tmp / "runs" / "fake_1b.json").read_text(encoding="utf-8"))
    check(len(run["rows"]) == 30, f"{len(run['rows'])} rows, expected 30")
    check(run["memory_bytes"] == 2_000_000_000 and run["quantization"] == "Q4_K_M", "memory or model info not recorded")
    system, think, options = FakeOllama.seen_system[-1]
    check(system == generate.SYSTEM and think is False and options["temperature"] == 0.0, "request settings wrong")
    s = summarise(run, questions, {c.id: c.text for c in chunks})
    check((s["right"], s["declined"]) == (0, 10), f"always-decline model scored {s['right']}/20, {s['declined']}/10")
    return "generate -> run file -> scorer works against a stand-in server (0/20, 10/10 as expected)"


def step_full_local(tmp):
    from lvc import generate
    from lvc.report import summarise

    chunks, questions = load()
    model = KNOWN["full_model"]
    old = generate.RUNS
    generate.RUNS = tmp / "runs"
    try:
        with redirect_stdout(io.StringIO()):
            generate.main(model)
    except OSError as e:
        raise Fail(f"Ollama not reachable on 127.0.0.1:11434 ({e})")
    finally:
        generate.RUNS = old
    run = json.loads((tmp / "runs" / (model.replace(":", "_") + ".json")).read_text(encoding="utf-8"))
    s = summarise(run, questions, {c.id: c.text for c in chunks})
    exp = KNOWN["scores"][model]
    got = {"found": s["found"], "right": s["right"], "declined": s["declined"], "wrong": s["wrong"]}
    check(got == exp, f"{model} rerun scored {got}, saved run {exp}")
    return f"{model} rerun on real Ollama matches the saved scores ({s['right']}/20, {s['declined']}/10, {s['total_s']:.0f} s)"


def step_full_retrieval(tmp):
    import subprocess

    r = subprocess.run([sys.executable, "-m", "lvc.check_retrieval"], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    check(r.returncode == 0, out[-1] if out else "check_retrieval failed")
    return out[-1]


def main():
    full = "--full" in sys.argv
    steps = [("compile", step_compile), ("units", step_units), ("scores", step_scores),
             ("generate", step_generate_offline)]
    if full:
        steps += [("local-rerun", step_full_local), ("retrieval", step_full_retrieval)]
    print(f"verify-repo: {ROOT}")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, fn in steps:
            try:
                print(f"  {name}: {fn(tmp)}")
            except Fail as e:
                print(f"verify: FAIL at {name}: {e}")
                sys.exit(1)
            except Exception as e:  # an unexpected crash is a failure of that step
                print(f"verify: FAIL at {name}: {type(e).__name__}: {e}")
                sys.exit(1)
    print("verify: PASS")


if __name__ == "__main__":
    main()
