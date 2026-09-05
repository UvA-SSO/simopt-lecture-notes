# CLAUDE.md

This file provides guidance to Claude when working in this repository, which is used
to organize lecture notes for a course as a [Jupyter Book](https://jupyterbook.org)
(v2, MyST-based) site.

## What this repo is

A Jupyter Book project where course lecture notes live as Jupyter notebooks, rendered
into a browsable, searchable book/site via MyST. There is no application code — content
is the product.

## Environment and commands

Dependencies are managed with `uv` (see `uv.lock`); Python `3.12` is pinned in
`.python-version`.

```sh
uv sync                                        # install dependencies into .venv
uv run jupyter book start --execute            # build, execute, and serve locally (reads myst.yml)
uv run jupyter book build --execute --html     # one-shot build (used for verification)
uv run pre-commit run --all-files              # run all lint/format/sync hooks
```

Notebooks contain **real, executed Python code** (pulp for optimization, `numpy`/`pandas`/
`scipy`/`matplotlib` for statistics and simulation) — the `--execute` flag is required or
`jupyter book` will render stale/no output. Execution is a mystmd Beta feature: it starts a
local Jupyter server per build (cached under `execute/`, gitignored). A broken code cell
fails the whole build.

There are no automated tests in this repo; a clean `--execute` build (or, if the sandbox's
network policy blocks `api.mystmd.org`'s template fetch, extracting and running each
script's code cells directly with `uv run python`) is the closest thing to one.

## Architecture: notebooks are paired with scripts via Jupytext

Every notebook in `notebooks/*.ipynb` has a paired plain-Python script in
`scripts/*.py` (percent format, `# %%` cells), linked via `jupytext` and configured in
`pyproject.toml`:

```toml
[tool.jupytext.formats]
"notebooks/" = "ipynb"
"scripts/" = "py:percent"
```

The pairing is enforced by the `jupytext --sync` pre-commit hook. Practical implications:

- When editing a notebook's content (including lecture notes), prefer editing the
  paired `.py` file in `scripts/` rather than hand-editing notebook JSON — the `.py`
  file is the diff-friendly source of truth.
- The filename stem must match between `notebooks/<name>.ipynb` and `scripts/<name>.py`.
- Do not manually edit outputs/metadata in the `.ipynb` beyond what jupytext sync would
  produce; let the sync hook reconcile them.
- After creating/editing a `.py` script, run `uv run jupytext --sync scripts/<name>.py`
  (or let the pre-commit hook do it) to generate/update the paired notebook.

## Organizing lecture notes

- One notebook per **topic** within a lecture, not one per lecture. Name
  notebooks/scripts `lectureN_topic-slug.ipynb` / `.py` (e.g.
  `lecture8_introduction.py`, `lecture8_linear-optimization.py`), with a short
  kebab-case slug for the topic.
- `project.toc` in `myst.yml` groups topic notebooks under a `title`-only parent
  entry per lecture (no `file`, just `children`) so the left-hand nav shows
  "Lecture N" as an expandable entry listing its topic notebooks — see below.
- Notebooks read as coherent, standalone material — no book chapter/section
  numbering, page numbers, or "Compiled source text" / "Source map" wrapper.
  Concretely:
  - Headings are plain topic names (`## Problem Formulation`, not `#### 6.1
    Problem formulation`); drop a subsection heading entirely if it would just
    restate the page's own H1 title.
  - Book "Exercise N.M" / "Example N.M" become MyST's native `{exercise}` /
    `{note} Example: ...` blocks (auto-numbered per page, no book numbers), each
    with a `:label:` (e.g. `ex-6-1`) for cross-referencing. `{proof:example}` is
    **not supported** by this project's mystmd setup (renders as a raw, broken
    directive) — use `{note}` for examples. Book "Box N.M. Title" becomes a
    `{note}` directive titled with just the descriptive title.
  - A directive's `:::{...}` opening and closing fence must stay within a single
    notebook cell (one `# %% [markdown]` block) — it cannot span across cells,
    so an image that was "inside" a book box must be embedded in the same cell,
    not split into its own markdown cell.
  - Cross-references to material in another topic notebook (or elsewhere on the
    same page) use a `(label)=` anchor before the target heading/paragraph and a
    `[text](lectureN_topic.ipynb#label)` (or `#label` for same-page) link —
    verified working across pages in this project. Pointers to material outside
    these notebooks (e.g. a book chapter not compiled here) are reworded away,
    not linked.
  - Each notebook ends with a `## References` section: a manual bullet list (no
    `.bib` file / MyST `{cite}` roles) starting with the Koole (2019) citation
    and the specific section(s) compiled on that page, followed by one entry per
    external work the page's text cites (replacing the book's own "[8]"-style
    bracket citations with author-year text, e.g. "Davenport & Harris (2007)").
- All Excel/R/AMPL content has been converted to Python. Optimization uses
  [pulp](https://coin-or.github.io/pulp/) (`pip install pulp[cbc]`, already a
  project dependency); statistics/simulation use `scipy.stats`, `numpy`,
  `pandas`, `matplotlib`. R-only base datasets (`eurodist`, `AirPassengers`,
  `beaver1`, `beaver2`, `Nile`) are one-time snapshots fetched from
  [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) (via
  `raw.githubusercontent.com`, not the blocked `.github.io` domain) and
  committed under `notebooks/data/*.csv` — read them with
  `pd.read_csv("data/<name>.csv")`, don't refetch at runtime.
- Figures (`![caption](images/...)`) are wrapped in MyST `{figure}`/`{table}`
  directives with a `:label:` (original book caption, no "Figure N.M:" numbering)
  so they're auto-numbered and cross-referenceable via `[](#label)` — this is a
  separate, verified-working mechanism from the code-cell one below.
- **Do not** try to label a live matplotlib plot's *output* as a numbered
  `{figure}` via mystmd's `#| label: cell-name` code-cell-tag convention +
  `` :::{figure} #cell-name ` `` — it renders fine in mystmd itself, but this
  project's pinned `ruff-format` (`ruff-pre-commit` rev `v0.6.9`, older than the
  `ruff` used elsewhere) rewrites `#| label: x` to `# | label: x` on every
  pre-commit run, silently breaking the cell tag. Just let the plot render as
  plain, unlabeled code-cell output, introduced by a descriptive sentence
  ("the figure below shows...") instead of a `[](#label)` cross-reference.
- Put all `import` statements for a notebook in its **first** code cell. `isort`
  runs with `--float-to-top` and will relocate any `import` found in a later
  cell up to the first one on the next pre-commit run, which silently breaks
  cell boundaries/comments placed next to it if it isn't already there.
- `mypy` type-checks each script as a single flat file (cell boundaries aren't
  scopes), so reusing a loop variable name (e.g. `i`) with an incompatible type
  in a later cell (e.g. as a dict string key after it was an `int` range index
  earlier) is a real type error — use distinct variable names per cell instead.

## Architecture: book structure via `myst.yml`

`myst.yml` is the single source of truth for the book's structure and site config:

- `project.toc` defines the table of contents — notebooks are only included in the
  built book if listed here (adding a notebook file alone is not enough). Structure:
  ```yaml
  toc:
    - file: index.md
    - title: Lecture 8        # no `file`: an expandable, non-clickable nav group
      children:
        - file: notebooks/lecture8_introduction.ipynb
          title: Introduction to Business Analytics
        - file: notebooks/lecture8_linear-optimization.ipynb
          title: Linear Optimization
  ```
- `site` configures the rendered site (theme, title, logo text).

When adding a new lecture topic notebook:

1. Create the paired `.py` file in `scripts/` (or `.ipynb` in `notebooks/`, then sync).
2. Add a corresponding entry under the right lecture's `children` list in
   `project.toc` in `myst.yml` (create the lecture's `title`-only entry if it's new).
3. Update `index.md`'s lecture/topic list to match.
4. Run `uv run jupyter book start` to verify it renders and appears in the nav.

Build output goes to `_build/` and the execution cache to `execute/` (both
gitignored) and should never be committed.

## Version control

Use Git to track progress as lecture notes are developed — commit regularly (e.g.
after finishing a lecture, or a meaningful chunk of one) rather than accumulating
large, unreviewed changes. This keeps history useful and makes it easy to see how
each lecture evolved.

## Pre-commit hooks

`.pre-commit-config.yaml` runs, in order: standard hygiene checks (trailing
whitespace, EOF fixer, TOML check, debug-statement check), `isort` (with `# %%` cell
markers treated as code boundaries, `--float-to-top`), `ruff` (lint + format), `mypy`,
then `jupytext --sync` last. `fail_fast: true` is set, so hooks stop at the first
failure — fix and re-run rather than expecting later hooks to also report.

Run `uv run pre-commit run --all-files` before committing lecture note changes.
