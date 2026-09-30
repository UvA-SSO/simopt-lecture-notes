# Changelog

The lecture notes are updated during the course. This page lists what changed in each published version, newest first. The version you are reading is shown at the bottom of every page.

Small fixes (typos, formatting) are grouped into a single line. Each entry links to the full list of changes on GitHub.

## 2026-09-30 21:31

- Lecture 10 is split into three pages. [Multi-Period Inventory Planning](notebooks/lecture10_multi-period.ipynb) and [Robust Regression](notebooks/lecture10_robust-regression.ipynb) each have their own page, and the new [Solvers and How to Help Them](notebooks/lecture10_solvers.ipynb) replaces Modeling Tools and Solvers. It covers solver quality and how much faster solvers have become, how to call another solver such as HiGHS, how to read a CBC solve log (incumbent, best bound and gap), and what to do when a solve takes too long: time limits, gap tolerances, warm starts and tighter formulations. Algebraic modeling languages keep a short note.
- [Robust Regression](notebooks/lecture10_robust-regression.ipynb) has new plots: the example data, a line that is not optimal with its errors, the optimal line, and quantile regression lines for $p = 0.5$ and $p = 0.9$. The quantile regression model is written out in full.
- To use HiGHS on your own computer, install PuLP with `pip install "pulp[highs]==3.3.2"` (see [PuLP version](index.md#pulp-version)). In Colab, HiGHS is already available.
- [Lecture 10 homework exercises](notebooks/lecture10_exercises.ipynb): exercise 2 is now about reading a solver log instead of modeling languages, exercise 3 is shorter, and exercise 1h uses the warehouse capacity $C_1$ instead of the rental cost $R_1$. [Multi-Period Inventory Planning](notebooks/lecture10_multi-period.ipynb) now says when an order arrives.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-09-29-1110...2026-09-30-2131)

## 2026-09-29 11:10

- The Lecture 9 applications ([Transportation and Transshipment](notebooks/lecture9_transportation.ipynb), [Set Covering and Shift Scheduling](notebooks/lecture9_covering.ipynb), [Machine Scheduling](notebooks/lecture9_machine-scheduling.ipynb)) and the Lecture 10 applications ([Multi-Period Inventory Planning](notebooks/lecture10_multi-period.ipynb), [Robust Regression](notebooks/lecture10_robust-regression.ipynb)) now follow the same steps as the slides: practical motivation, the general model built up step by step, the example model written out, and the solution in pulp, followed by extensions.
- [Machine Scheduling](notebooks/lecture9_machine-scheduling.ipynb) first builds the scheduling model, introducing order variables and big M where the model needs them, and then treats big M, conditional and either/or constraints in general in a separate section.
- [Integer Optimization](notebooks/lecture8_integer-optimization.ipynb): the knapsack problem is introduced in the same way as the other applications, followed by branch and bound for the knapsack example in its own section. The integer lattice plot now shows the optimal-profit line.
- The [Lecture 9 homework exercises](notebooks/lecture9_exercises.ipynb) are reworked: every decision variable is defined explicitly, the exercise 3 network is redrawn, the couples exercise has numbers, and the pulp code exercise has hints. The class-scheduling exercise and the further exercises are removed.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-09-27-0840...2026-09-29-1110)

## 2026-09-27 08:40

- Solving models with pulp works in Google Colab again. Since 24 September, the solve cells failed there with a CBC error; they now use `pulp.PULP_CBC_CMD(msg=False)` again, as before.
- PuLP 4.0 has been released, and the code in these notes does not run on it. These notes stay with PuLP 3.3.2: see [PuLP version](index.md#pulp-version) for how to install it on your own computer and where to find its documentation.
- [Integer Optimization](notebooks/lecture8_integer-optimization.ipynb): a new, more detailed explanation of branch and bound with step-by-step trees, a new exercise on branching, and the knapsack problem worked out on one example, including solving its LO relaxation by hand. The [lecture 8 homework solutions](notebooks/lecture8_exercises.ipynb) now follow this explanation step by step.
- New exam-style exercises with questions about pulp code, to be answered on paper: [lecture 8](notebooks/lecture8_exercises.ipynb#hw-8-5), [lecture 9](notebooks/lecture9_exercises.ipynb#hw-9-6) and [lecture 10](notebooks/lecture10_exercises.ipynb#hw-10-3).
- Graphical representations of optimization problems are interactive (hover over a point to see its coordinates and objective value), and figures and link previews read well in dark mode. Models are written with one constraint per line, figure captions name the exercise they belong to, this changelog and the version line at the bottom of every page are new, and a few typos and formatting issues are fixed.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/0ced1c0...2026-09-27-0840)
