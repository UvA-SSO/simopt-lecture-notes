# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lecture 10: Modeling Tools and Solvers

# %% [markdown]
# So far we discussed solving LO and ILO problems, and the engines to solve them. However, we should realize that optimization specialists spend most of their time on modeling. Modeling is the translation of a real-life problem into a mathematical description that can be used to solve the problem. See [](#fig-modeling-steps) for the steps in modeling. The time spent on modeling can be greatly reduced by using an appropriate modeling tool or language. As such, the existence of these modeling tools is considered to be of equal importance as the engines used to solve the models. To learn a modeling tool or language requires some time, but this easily pays off if you often build models for optimization problems.

# %% [markdown]
# :::{figure} images/lecture10_fig6.10.png
# :label: fig-modeling-steps
#
# Modeling steps.
# :::

# %% [markdown]
# The best known algebraic modeling languages (AMLs) are AIMMS, AMPL, GAMS, LINDO, and MPL. Some of these languages are part of an integrated development environment (IDE) that simplifies the modeling even further: the problem entry in these AMLs is quite similar to mathematical notation. Many of the tools and engines have free educational licenses, which makes it possible for students to learn and experiment.
#
# In earlier notebooks we already used [pulp](https://coin-or.github.io/pulp/), a Python library that gives us the same benefits as a dedicated AML — and more, because it is regular Python underneath. This notebook goes deeper into pulp itself: how to structure larger models, which solver to use and how to read its output, warm starting, and how to use a GenAI assistant productively when a model doesn't behave as expected.
#
# The AMLs and associated IDEs allow experienced modelers to model and solve problems they encounter. However, engines can also be built into software dedicated to solve a particular class of problems such as navigation software. These types of problem-specific tools are called decision support systems (DSSs). They are geared towards a different class of users. While AMLs (and pulp) are used by experienced data scientists and OR consultants, DSSs are most often used by planners with domain knowledge, but with less or no background in optimization and modeling.

# %% [markdown]
# ## Installing pulp
#
# pulp is a regular Python package:
#
# ```
# pip install pulp[cbc]
# ```
#
# The `[cbc]` extra bundles a ready-to-use build of the open-source [CBC](https://github.com/coin-or/Cbc) solver, so no separate solver installation is needed to get started (see the [installation instructions](https://coin-or.github.io/pulp/main/installing_pulp_at_home.html) for details and alternatives). We already installed it this way for this course's environment.

# %% [markdown]
# ## Structuring Code: Data versus Model
#
# A pattern we have used throughout the previous two notebooks, worth making explicit: keep the problem *data* (the numbers that describe a particular instance) completely separate from the *model* (the pulp objects that describe the optimization problem in general). Concretely, structure your code as
#
# 1. plain Python data structures (lists, dicts, `pandas` DataFrames, ...) holding the instance data;
# 2. a `pulp.LpProblem`, built from that data using decision variables, an objective, and constraints.
#
# This is exactly what an AML gives you (a model file separate from a data file), except here both live in the same Python session, so nothing needs to be exported and re-imported between them. If tomorrow you need to solve the same *kind* of problem for a different instance, only the data changes; the model-building code is reused unchanged.
#
# Let's use the single-machine scheduling problem from [Advanced Modeling](lecture9_advanced-modeling.ipynb) as a running example. The data:

# %%
import pulp

jobs = ["Job A", "Job B", "Job C"]
duration = [10, 11, 15]  # duration[i]: duration of jobs[i]
n = len(jobs)

# %% [markdown]
# ## Decision Variables
#
# We introduce decision variable $x_i$, the starting time of job $i$, for $i = 1, \dots, n$. In pulp:

# %%
x = [pulp.LpVariable(name=f"x_{i}", lowBound=0, cat="Continuous") for i in range(n)]
print(type(x[1]), x[1])

# %% [markdown]
# `lowBound` sets the lower bound (default $-\infty$); an `upBound` can be added similarly (default $+\infty$). `cat` sets whether the variable is `"Continuous"` (the default), `"Integer"`, or `"Binary"`. `x` is a plain Python `list`, so `x[i]` refers to $x_{i+1}$ (the $+1$ because Python counts from 0) — nothing pulp-specific about that indexing, it is just how Python lists work.
#
# We could equally well use a dictionary keyed by job name, which reads more naturally once there are several groups of decision variables to keep apart:

# %%
x_dict = {jobs[i]: pulp.LpVariable(name=f"x_{i}", lowBound=0) for i in range(n)}
print(x_dict["Job B"])

# %% [markdown]
# pulp also provides a shorthand for building such a dictionary directly from a list of keys:

# %%
x_dict_pulp = pulp.LpVariable.dicts(name="x", indices=jobs, lowBound=0)
print(x_dict_pulp["Job A"])

# %% [markdown]
# ### Double-Indexed Decision Variables
#
# Let binary decision variable $y_{ij}$ be 1 if job $i$ goes before job $j$, for $i, j = 1, \dots, n$. This is just a list of lists (or a nested dict):

# %%
y = [[pulp.LpVariable(name=f"y_{i},{j}", cat="Binary") for j in range(n)] for i in range(n)]
print(y[1][2])

# %% [markdown]
# ## Objective and Constraints
#
# Both the objective and the constraints are added to an `LpProblem` with `+=`: an expression *without* a comparison (`==`, `<=`, `>=`) is registered as the objective, an expression *with* one becomes a constraint. Suppose we want to minimize the total finish time $\sum_{i=1}^{n} (x_i + d_i)$, where $d_i$ is the duration of job $i$:

# %%
schedule = pulp.LpProblem(name="schedule", sense=pulp.LpMinimize)
schedule += pulp.lpSum(x[i] + duration[i] for i in range(n)), "sum_of_finish_times"

# %% [markdown]
# The optional string after the comma names the objective (or constraint) — useful when printing the problem or reading a solver log. Constraints are added the same way, for example that no job may start after time 10:

# %%
for i in range(n):
    schedule += x[i] <= 10, f"job_{i}_starts_before_10"

print(schedule)

# %% [markdown]
# Printing an `LpProblem` shows every variable, the objective, and every named constraint — a good habit whenever a model doesn't behave as expected, especially for a small version of the instance.

# %% [markdown]
# (solving-and-reading-the-log)=
# ## Solving and Reading the Solver Log
#
# `problem.solve()` solves the model with pulp's default solver (CBC). Passing `msg=True` shows CBC's own branch-and-bound log — normally suppressed, but worth seeing at least once to know what it means:

# %%
schedule.solve(pulp.PULP_CBC_CMD(msg=True))
print("status:", pulp.LpStatus[schedule.status])
print("finish times:", [xi.value() for xi in x])

# %% [markdown]
# In the log above, the line reporting the MPS file size (`Problem MODEL has ... rows, ... columns`) confirms how many constraints/variables CBC actually sees after simplification. Every time CBC improves the best known integer solution during branch-and-bound (see [Integer Optimization](lecture8_integer-optimization.ipynb) for what that means) it reports the new objective value and the remaining optimality *gap*: the guaranteed distance between the best solution found so far and the best possible bound. A run that finishes with `Optimal` and gap 0% has proven optimality; a run that stops early (e.g. due to a time limit) instead reports the best solution found together with the gap still open at that point — useful to know if you can afford to keep waiting for a smaller gap.
#
# For everyday use `msg=False` is usually what you want, so the solver's log doesn't clutter your output — that's what we used in the previous two notebooks.

# %% [markdown]
# ## Choosing a Solver
#
# pulp can call several different solvers without changing the model itself — only the line that calls `.solve()` changes. To see which solvers are available in the current environment:

# %%
print(pulp.listSolvers(onlyAvailable=True))

# %% [markdown]
# `PULP_CBC_CMD` is the default, bundled solver we have used throughout. If a faster open-source alternative such as [HiGHS](https://highs.dev/) is installed (`pip install pulp[highs]` or similar, depending on platform), it can be used the same way: `schedule.solve(pulp.getSolver("HIGHS", msg=False))`.
#
# The best performance on hard, large-scale problems is generally obtained with commercial solvers such as Gurobi, FICO Xpress, and CPLEX. pulp can call those too, if installed and licensed, without any change to the model:
#
# ```python
# solver = pulp.getSolver("GUROBI")
# schedule.solve(solver)
# ```
#
# Without installing any solver locally, an alternative is to export the model to a standard `.mps` file and submit it to the free [NEOS Server](https://neos-server.org/neos/solvers/index.html), which offers a range of solvers including commercial ones:

# %%
schedule.writeMPS("schedule.mps")

# %% [markdown]
# If you are concerned about sharing your own variable/constraint names (which may reveal something about the underlying data) with a third-party server, pulp can anonymize them on export:

# %%
mapping = schedule.writeMPS("schedule_renamed.mps", rename=1)
print("variable name mapping:", mapping[1])

# %% [markdown]
# :::{exercise}
# :label: ex-10-1
#
# Re-solve [the knapsack problem](lecture8_integer-optimization.ipynb) with `msg=True` and identify, in the log, the step at which CBC first proves optimality (gap reaches 0%).
# :::

# %% [markdown]
# ## Warm Starting
#
# Often you need to re-solve a problem that is very similar to one you already solved — for example the same knapsack with one extra item, or the same schedule with one job's duration updated. Rather than starting from scratch, pulp can pass CBC an initial solution to *warm start* from: set each variable's initial value with `.setInitialValue(...)`, then solve with `warmStart=True`.

# %%
revenue = [60, 60, 40, 10, 20, 10, 3]
weight = [3, 5, 4, 1.4, 3, 3, 1]


def build_knapsack(capacity: float) -> tuple[pulp.LpProblem, list[pulp.LpVariable]]:
    problem = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
    items = [pulp.LpVariable(name=f"x_{i+1}", cat="Binary") for i in range(len(revenue))]
    problem += pulp.lpSum(revenue[i] * items[i] for i in range(len(revenue)))
    problem += pulp.lpSum(weight[i] * items[i] for i in range(len(revenue))) <= capacity
    return problem, items


knapsack_11, items_11 = build_knapsack(capacity=11)
knapsack_11.solve(pulp.PULP_CBC_CMD(msg=False))
print("capacity 11:", [xi.value() for xi in items_11], knapsack_11.objective.value())

knapsack_12, items_12 = build_knapsack(capacity=12)
for item, item_prev in zip(items_12, items_11):
    item.setInitialValue(item_prev.value())
knapsack_12.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))
print("capacity 12, warm started:", [xi.value() for xi in items_12], knapsack_12.objective.value())

# %% [markdown]
# For a knapsack this small the effect on solve time is not measurable, but for large ILO models that are re-solved repeatedly with small data changes (e.g. inside a simulation-optimization loop, see [Simulation Optimization](lecture13_simulation-optimization.ipynb)), starting from a known-good solution can noticeably reduce the number of branch-and-bound nodes CBC has to explore.
#
# :::{exercise}
# :label: ex-10-2
#
# Warm start [the shift-scheduling exercise](lecture9_ilo-applications.ipynb#ex-6-13) from the previous day's optimal schedule when one interval's demand changes by a small amount. Compare the reported number of explored nodes (visible in the `msg=True` log) with and without warm starting.
# :::

# %% [markdown]
# ## Debugging with GenAI
#
# A GenAI assistant (such as Claude or ChatGPT) can be a genuinely useful pulp pair-programmer, provided you use it well:
#
# - Paste the actual pulp code and the actual error message or solver log, not a vague description of the problem — GenAI models are much better at spotting a missing `<=`, a wrong index, or an infeasible combination of constraints when they can see the real code.
# - Always re-check a GenAI-suggested model against the original problem data and formulation yourself: it can misread a constraint's direction, silently change what a variable represents, or introduce a subtly wrong index range, and pulp will happily build and solve the wrong model without complaint.
# - `print(problem)` (as in [Solving and Reading the Solver Log](#solving-and-reading-the-log) above) before and after any GenAI-suggested change is a quick way to see exactly what changed.
# - For a status that is `"Infeasible"` or `"Unbounded"` (see [Linear Optimization](lecture8_linear-optimization.ipynb)), a GenAI assistant can be helpful for spotting the conflicting or missing constraint, especially in a larger model — but it needs the full model text to do so, not just a description of the symptom.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.6 "Modeling Tools." (Modeling-language content extended with pulp's own solver, warm-starting, and debugging features.)
# - PuLP documentation: https://coin-or.github.io/pulp/
# - `Course Materials/pulp_tutorial.py` (this course's own PuLP tutorial, by Joost Berkhout).
