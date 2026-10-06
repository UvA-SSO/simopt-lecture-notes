# Changelog

The lecture notes are updated during the course. This page lists what changed in each published version, newest first. The version you are reading is shown at the bottom of every page.

Small fixes (typos, formatting) are grouped into a single line. Each entry links to the full list of changes on GitHub.

## 2026-10-06 07:43

- Lecture 12 is rewritten to follow the slides and split into five pages, replacing Simulation. [Variability (Recap)](notebooks/lecture12_variability-recap.ipynb) is now a short theory-only recap of the probability and statistics that Lectures 12 and 13 use. [Why Simulate?](notebooks/lecture12_why-simulation.ipynb) describes the simulation setting and the flaw of averages, [Sampling a Random Variable](notebooks/lecture12_sampling.ipynb) the inverse transform method and sampling with a numpy generator and a seed, [Monte Carlo Simulation](notebooks/lecture12_monte-carlo.ipynb) the project planning example with numpy arrays, and [Discrete-Event Simulation](notebooks/lecture12_discrete-event-simulation.ipynb) a queue in Python and model validation.
- You need to be able to read the code of a discrete-event simulation, not write one yourself.
- [Lecture 12 homework exercises](notebooks/lecture12_exercises.ipynb): two new exercises ask you to read a numpy Monte Carlo simulation of a project and the code of an $(s, S)$ inventory simulation.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-10-05-0807...2026-10-06-0743)

## 2026-10-05 08:07

- Lecture 11 is rewritten to follow the slides and split into five pages, replacing Algorithms and Heuristics and Complexity. [Algorithms and Their Characteristics](notebooks/lecture11_algorithms.ipynb) explains how the algorithms are written down and compares them (dedicated or general, polynomial or not, exact or heuristic). [Shortest Path](notebooks/lecture11_shortest-path.ipynb), [Maximum Flow](notebooks/lecture11_maximum-flow.ipynb) and [Traveling Salesman Problem](notebooks/lecture11_tsp.ipynb) each solve an example with an (I)LO model in pulp and with a dedicated algorithm. [Complexity and Heuristics](notebooks/lecture11_complexity-heuristics.ipynb) covers P and NP-complete problems and the 2-opt heuristic.
- Dijkstra's algorithm, the Ford-Fulkerson algorithm and 2-opt are each worked out step by step on the slides' example network, with a figure per iteration and short Python code that follows the algorithm line by line. You need to be able to read this code, not write it yourself.
- [Lecture 11 homework exercises](notebooks/lecture11_exercises.ipynb): the graphs are redrawn, and the new [exercise 6](notebooks/lecture11_exercises.ipynb#hw-11-6) asks you to trace a piece of code for the traveling salesman problem by hand, analyze its running time and improve it.
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-10-02-2036...2026-10-05-0807)

## 2026-10-02 20:36

- Simulation has a new section, [Simulating with numpy Arrays](notebooks/lecture12_monte-carlo.ipynb#numpy-arrays): why the simulations work with arrays instead of `for` loops, with a timing comparison, and a table of the numpy features the simulations in Lectures 12 and 13 use.
- [Simulation Optimization](notebooks/lecture13_ranking-and-selection.ipynb#option-2): in ranking and selection, the budget that remains after the first round is $m - k m_0$ (it said $m - m_0$).
- [Full diff on GitHub](https://github.com/UvA-SSO/simopt-lecture-notes/compare/2026-09-30-2131...2026-10-02-2036)

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
