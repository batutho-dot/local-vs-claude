# Generate

**What it does:** answers the 30 test questions with one local Ollama model, on the 5 passages
saved from rag-eval's Claude run, and writes `runs/<model>_<tag>.json` with answers, seconds,
token counts and memory.

**How to reach it:** `python run.py generate <model> [tag]`, `python run.py repeat <tag>...` (one model at a time, waits for free RAM), `python run.py all`.

**How verify drives it:** `generate` step points `lvc.generate` at a stand-in Ollama server
on a free local port that declines everything, writes to a temp folder, and checks 30 rows,
recorded memory and quantization, the system prompt sent, `think: false` for a thinking
model, temperature 0, and the scorer result 0/20, 10/10. `--full` reruns `llama3.2:3b` on
the real server and compares its scores with `verify_known_good.json`.

**End state:** a run file whose scores match the known-good ones.

**Gotchas:** `--full` needs `ollama serve` running and the model pulled; it fails with
"Ollama not reachable" otherwise. Temperature 0 made the rerun identical on 2026-11-19; a
new Ollama or GPU driver version could change answers, which would show up as a FAIL here.

The memory guard (refuse when another model is loaded, unload after the run) is driven by the stand-in server: it reports a model only while one is loaded, and the step fails if the model is still loaded at the end.
