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
uv sync                    # install dependencies into .venv
uv run jupyter book start  # build and serve the book locally (reads myst.yml)
uv run pre-commit run --all-files   # run all lint/format/sync hooks
```

There are no automated tests in this repo.

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

Build output goes to `_build/` (gitignored) and should never be committed.

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
