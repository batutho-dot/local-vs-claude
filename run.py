"""local-vs-claude: rag-eval's test, answered by local models instead of Claude.

    python run.py                  print the results table from the saved runs (no model needed)
    python run.py download         fetch the three EU texts from EUR-Lex into data/
    python run.py generate <model> answer the 30 questions with a local Ollama model, e.g. llama3.2:3b
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

        generate.main(sys.argv[2])
    elif cmd == "all":
        from lvc import generate, report
        from pipeline import download

        download.main()
        for m in MODELS:
            generate.main(m)
        report.main()
    else:
        sys.exit(__doc__)
