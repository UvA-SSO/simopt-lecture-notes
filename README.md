# Simulation and Optimization — Lecture Notes

Lecture notes for the Simulation and Optimization part of the course Statistics,
Simulation and Optimization (SSO), published as a
[Jupyter Book](https://jupyterbook.org) (v2, MyST-based) site. Content lives as
paired Jupyter notebooks (`notebooks/`) and plain-Python scripts (`scripts/`).

## Prerequisites

- Python `3.12` (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running commands

## Install

```sh
uv sync
```

This creates a `.venv` and installs everything listed in `uv.lock`, including
[pulp](https://coin-or.github.io/pulp/) (with the bundled CBC solver) and the
usual scientific-Python stack (`numpy`, `pandas`, `scipy`, `matplotlib`).

## Run the book locally

```sh
uv run jupyter book start --execute
```

This builds the book, **executes every notebook**, and serves it locally with
live reload (reads configuration from `myst.yml`). The `--execute` flag is
required — without it, code-cell outputs are stale or missing. A local
Jupyter server is started under the hood; execution results are cached in
`execute/` (gitignored).

For a one-shot build without serving (e.g. to verify everything renders
before committing):

```sh
uv run jupyter book build --execute --html
```

Build output goes to `_build/` (gitignored).

## Development workflow

Every notebook in `notebooks/*.ipynb` is paired with a script in
`scripts/*.py` (percent format, `# %%` cells) via
[Jupytext](https://jupytext.readthedocs.io/). **Edit the `.py` file**, not the
notebook JSON — it's the diff-friendly source of truth. After editing, sync
the pair:

```sh
uv run jupytext --sync scripts/<name>.py
```

Before committing, run all lint/format/sync checks:

```sh
uv run pre-commit run --all-files
```

This runs, in order: hygiene checks, `isort`, `ruff` (lint + format), `mypy`,
then `jupytext --sync`. Hooks stop at the first failure — fix and re-run.

There are no automated tests in this repo; a clean `--execute` build is the
closest thing to one.

## Deployment

`.github/workflows/deploy-pages.yml` runs on every push and pull request: it
runs the same lint/format/sync checks as `pre-commit` and builds the book
(with execution), so a broken notebook or failing check shows up directly on
a PR. Only on a push to `main`, after that build succeeds, does it publish
the result to GitHub Pages. Enable Pages once, per repo: **Settings → Pages →
Source → GitHub Actions**. To make the check block merging until it passes,
also add a branch protection rule for `main` under **Settings → Branches**
requiring the `build` check to succeed.

## Repository structure

```
notebooks/        Jupyter notebooks (generated/synced from scripts/, do not hand-edit)
scripts/           Paired plain-Python source (edit these)
notebooks/data/    Small committed datasets (CSV)
notebooks/images/  Figures referenced by the notebooks
myst.yml           Book structure (table of contents) and site config
```

Adding a new lecture topic notebook, coding conventions, and further details
are documented in `CLAUDE.md`.
