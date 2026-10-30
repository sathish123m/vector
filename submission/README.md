# Submission package — Week 1–5

| File | Purpose |
|---|---|
| `report.pdf` | Self-contained report for submission (no title page, per request) |
| `report.md` | Editable Markdown source of the same report |
| `oral_script.md` | Timed 10-minute script, delivery notes, and Q&A preparation |
| `assets/` | Figures used in the report |

Regenerate everything with:

```bash
.venv/bin/python scripts/make_submission_assets.py
.venv/bin/python scripts/build_submission.py
```

The report and script are generated from `scripts/submission_content.py`. Measured
numbers come from `reports/phase1/e1_sift100k/result.json` and the committed
run logs; the 28 October rerun is referenced as a reproducibility check, not
as the primary evidence.
