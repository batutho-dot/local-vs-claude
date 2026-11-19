# Architecture

## (a) Structure

The repo adds a thin layer (`lvc/`) that sends rag-eval's questions to a local model and
scores the answers with rag-eval's own code, which is copied in unchanged.

```mermaid
flowchart LR
    subgraph repo["local-vs-claude"]
        RUN["run.py<br/><i>commands, model list</i>"]
        GEN["lvc/generate.py<br/><i>fixed passages, same prompt,<br/>time, memory</i>"]
        REP["lvc/report.py<br/><i>three scores, table, misses</i>"]
        CHK["lvc/check_retrieval.py<br/><i>rerun search, compare ids</i>"]
        VER["verify.py<br/><i>stand-in server, known-good scores</i>"]
        RUNS[("runs/*.json<br/><i>3 runs per model: answers,<br/>seconds, memory</i>")]
        CLA["lvc/claude.py<br/><i>Haiku repeats, 0.50 USD stop</i>"]
        subgraph reused["pinned copy of rag-eval (commit e2cfdad)"]
            PIPE["pipeline/<br/><i>download, chunk, hybrid search</i>"]
            SCORE["rageval/report.py<br/><i>rule-based scorer</i>"]
            TEST[("testset.json<br/><i>20 answerable, 10 unanswerable</i>")]
            REF[("reference/haiku_hybrid.json<br/><i>Claude answers, retrieved ids</i>")]
        end
    end
    OLLAMA["Ollama on this PC<br/><i>127.0.0.1:11434, local models, GPU</i>"]
    CLAUDE["Claude API<br/><i>Haiku 5.5 repeat runs,<br/>run 1 saved by rag-eval</i>"]

    RUN --> GEN
    RUN --> REP
    GEN -- "/api/chat, /api/ps" --> OLLAMA
    GEN --> RUNS
    GEN --> REF
    GEN --> TEST
    GEN --> PIPE
    REP --> RUNS
    REP --> REF
    REP --> SCORE
    CHK --> PIPE
    CHK --> REF
    VER --> GEN
    VER --> REP
    CLAUDE -.->|"run 1, in rag-eval"| REF
    RUN --> CLA
    CLA --> CLAUDE
    CLA --> RUNS
```

## (b) One evaluation run

Every question gets the same five passages that Claude got, only the model that writes the
answer changes, and one script scores all answers the same way.

```mermaid
flowchart LR
    Q["Question<br/><i>testset.json, 30 items</i>"]
    R["Fixed retrieval<br/><i>5 chunk ids saved<br/>from the Claude run</i>"]
    P["Prompt<br/><i>rag-eval's system prompt,<br/>numbered passages</i>"]
    M{"Model switch"}
    L["Local model via Ollama<br/><i>llama3.2:3b, qwen3.5:4b,<br/>ministral-3:8b, temperature 0</i>"]
    H["Claude Haiku 5.5<br/><i>saved answers,<br/>reference only</i>"]
    S["Scorer<br/><i>passage found, required terms,<br/>NOT IN SOURCES</i>"]
    T["Results table<br/><i>three scores, seconds, memory</i>"]

    Q --> R --> P --> M
    M -->|"local"| L --> S
    M -->|"reference"| H --> S
    S --> T
```

## Where each box lives

| Box | File:line | What it does |
|---|---|---|
| Commands, model list | `run.py:15` | `MODELS`, and the `report`, `download`, `generate`, `claude`, `repeat`, `all` commands |
| Fixed retrieval | `lvc/generate.py:170` | loops over the saved Claude rows and takes their `retrieved` chunk ids (line 172) |
| Prompt | `lvc/generate.py:35`, `lvc/generate.py:44` | system prompt and user prompt, copied word for word from rag-eval |
| Local model call | `lvc/generate.py:70` | `/api/chat` with temperature 0, context 8,192, `think: false` for thinking models (line 79) |
| Memory | `lvc/generate.py:178` | reads the loaded model's size and GPU share from `/api/ps` |
| Memory guard | `lvc/generate.py:119`, `lvc/generate.py:151` | refuses to load if another model is loaded or free RAM is below the model's memory plus 3 GB; unloads after the run |
| Claude repeat runs | `lvc/claude.py:42` | Haiku 5.5 on the same passages and prompt, with a 0.50 USD total stop |
| Claude reference | `lvc/generate.py:26` | `reference/haiku_hybrid.json`, rag-eval's saved hybrid run |
| Scorer | `rageval/report.py:39` | `script_grades`: passage found, required terms present, declined |
| Results table | `lvc/report.py:56` | one row per model: mean over runs and the min to max range, then each run's misses |
| Hybrid search (rerun check) | `pipeline/hybrid.py:17`, `lvc/check_retrieval.py:17` | TF-IDF plus bge-small, fused by rank; compared with the saved ids |
