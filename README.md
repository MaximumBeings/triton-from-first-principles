# Triton From First Principles

A 22-chapter book about the classic Triton kernel language, tensor contracts,
autograd integration, measurement, attention and financial-computing examples.

[Published book](https://maximumbeings.github.io/triton-from-first-principles/)
· [Installation](docs/getting-started.md)
· [Review and execution scope](docs/review-and-testing.md)

The printed historical outputs are interpreter references. They do not establish
compiled GPU correctness or performance. The current edition exports all 84
complete reference files, records their identities and adds 55 original SVGs, including banners and worked diagrams for all four appendices.
The separate GPU smoke suite passed 12 H100 cases. Changed chapter references now have explicit pending revalidation status; an archived zero exit is not current-source numerical validation.

## Build and verify without a GPU

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements.txt
python tools/verify_book.py
python tools/verify_execution.py
python tools/verify_svg_readability.py
mkdocs build --strict
python tools/verify_site_links.py
mkdocs serve
```

`tools/create_diagrams.py` regenerates the embedded-font vector artwork using
the included Inter font and its SIL Open Font License. All concept labels fit
their canvases. The same-page diagram dialog supports fit width, fit whole,
Escape/Close and navigation cleanup.

After editing an explicitly titled complete reference in a chapter, run
`python tools/sync_examples.py`. CI rejects drift between inline code, downloads
and the source manifest. Python parsing is not a Triton compilation test.

Pushes to main run offline checks and deploy through `.github/workflows/deploy.yml`.
No GPU is provisioned by that workflow. Preserve real execution records when
GPU testing becomes available; never manufacture outputs or relabel interpreter
results as GPU runs.
