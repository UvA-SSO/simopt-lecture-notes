# Changelog

The lecture notes are updated during the course. This page lists what changed in each published version, newest first. The version you are reading is shown at the bottom of every page.

Small fixes (typos, formatting) are grouped into a single line. Each entry links to the full list of changes on GitHub.

## 2026-09-27 08:40

- Solving models with pulp works in Google Colab again. Since 24 September, the solve cells failed there with a CBC error; they now use `pulp.PULP_CBC_CMD(msg=False)` again, as before.
- PuLP 4.0 has been released, and the code in these notes does not run on it. These notes stay with PuLP 3.3.2: see [PuLP version](index.md#pulp-version) for how to install it on your own computer and where to find its documentation.
- [Integer Optimization](notebooks/lecture8_integer-optimization.ipynb): a new, more detailed explanation of branch and bound with step-by-step trees, a new exercise on branching, and the knapsack problem worked out on one example, including solving its LO relaxation by hand. The [lecture 8 homework solutions](notebooks/lecture8_exercises.ipynb) now follow this explanation step by step.
- New exam-style exercises with questions about pulp code, to be answered on paper: [lecture 8](notebooks/lecture8_exercises.ipynb#hw-8-5), [lecture 9](notebooks/lecture9_exercises.ipynb#hw-9-7) and [lecture 10](notebooks/lecture10_exercises.ipynb#hw-10-3).
- Graphical representations of optimization problems are interactive (hover over a point to see its coordinates and objective value), and figures and link previews read well in dark mode. Models are written with one constraint per line, figure captions name the exercise they belong to, this changelog and the version line at the bottom of every page are new, and a few typos and formatting issues are fixed.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/0ced1c0...2026-09-27-0840)
