# Changelog

The lecture notes are updated during the course. This page lists what changed in each published version, newest first. The version you are reading is shown at the bottom of every page.

Small fixes (typos, formatting) are grouped into a single line. Each entry links to the full list of changes on GitHub.

## 2026-09-26 15:18

- Solving models with pulp works in Google Colab again. Since 24 September, the solve cells failed there with a CBC error; they now use `pulp.PULP_CBC_CMD(msg=False)` again, as before.
- PuLP 4.0 has been released, with different code. These notes stay with PuLP 3.3.2: see [PuLP version](index.md#pulp-version) for how to install it on your own computer.
- [Integer Optimization](notebooks/lecture8_integer-optimization.ipynb): a new, more detailed explanation of branch and bound with step-by-step trees, a new exercise on branching, and the knapsack problem worked out on one example, including solving its LO relaxation by hand.
- [Lecture 8 exercises](notebooks/lecture8_exercises.ipynb): the homework solutions follow the branch and bound explanation step by step, with new figures.
- Graphical representations of optimization problems are interactive (hover over a point to see its coordinates and objective value), and all figures read well in dark mode. Models are written with one constraint per line, figure captions name the exercise they belong to, and a few typos and formatting issues are fixed.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-09-25-1428...2026-09-26-1518)

## 2026-09-25 14:28

- Started this changelog and added the version line at the bottom of every page.
- [Full history on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/commits/main)
