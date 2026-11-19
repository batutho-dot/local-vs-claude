"""local-vs-claude: rag-eval's test, answered by local models instead of Claude.

    python run.py                  print the results table from the saved runs (no model needed)
    python run.py download         fetch the three EU texts from EUR-Lex into data/
    python run.py generate <model> [tag]  answer the 30 questions with a local Ollama model,
                                   e.g. llama3.2:3b r2 (writes runs/<model>_<tag>.json)
    python run.py claude <tag>     repeat run with Claude Haiku 5.5 (costs about 0.01 USD; needs a key)
    python run.py repeat <tag>...  run every model in MODELS once per tag, one model at a time,
                                   waiting (up to 3 h per load) until RAM is free enough
    python run.py all              download, then generate with every model in MODELS, then report
"""

import sys

MODELS = ["llama3.2:3b", "qwen3.5:4b", "ministral-3:8b"]

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        from lvc import report

        report.main()
    elif cmd == "download":
        from pipeline import download

        download.main()
    elif cmd == "generate":
        from lvc import generate

        generate.main(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "r1")
    elif cmd == "claude":
        from lvc import claude

        claude.main(sys.argv[2])
    elif cmd == "repeat":
        from lvc import generate

        for tag in sys.argv[2:]:
            for m in MODELS:
                if (generate.RUNS / (m.replace(":", "_") + f"_{tag}.json")).exists():
                    continue  # already done
                if not generate.wait_for_ram(m):
                    sys.exit(f"not enough free RAM for {m} after 3 h; stopped before {m} {tag}")
                generate.main(m, tag)
    elif cmd == "all":
        from lvc import generate, report
        from pipeline import download

        download.main()
        for m in MODELS:
            generate.main(m)
        report.main()
    else:
        sys.exit(__doc__)
