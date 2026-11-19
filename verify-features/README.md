# Feature map

What `verify.py` checks, one file per feature.

| Feature | File | Checked by |
|---|---|---|
| Answer questions with a local model | `generate.md` | `generate` (stand-in server), `local-rerun` (`--full`) |
| Results table from saved runs | `report.md` | `scores` |
| Fixed retrieval | `retrieval.md` | `scores` (chunk count), `retrieval` (`--full`) |
