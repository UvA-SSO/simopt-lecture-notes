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
# ---
# description: "Which solvers exist, how to read a solver log (incumbent, best bound, gap), how to help a slow solve, and how GenAI is used in optimization."
# thumbnail: null
# ---
# # Lecture 10: Solvers and How to Help Them
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_solvers.ipynb)

# %% [markdown]
# So far we have written models in pulp and called `.solve(...)`, and the answer came
# back within a fraction of a second. For larger ILO models this is no longer true: the
# solver can run for minutes or hours, and it is useful to know what it is doing in the
# meantime. This notebook first compares the solvers that are available, then shows how
# to read the log that a solver writes while it works, and what you can do when a solve
# takes too long. It ends with the ways in which generative AI (GenAI) is used in
# optimization. The applications of this lecture are in
# [Multi-Period Planning and Robust Regression](lecture10_applications.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - compare open-source and commercial solvers, and explain where their speed comes from;
# - read a solver log: find the incumbent, the best bound and the gap, and tell whether
#   finding a good solution or proving its optimality is the hard part;
# - help a slow solve with a time limit or gap tolerance, a warm start, or a tighter
#   formulation;
# - describe three ways in which GenAI is used in optimization.

# %% [markdown]
# ## Solvers
#
# The solver is the software that does the optimizing; pulp builds the model and passes
# it to a solver. Roughly, there are two groups:
#
# | | examples | notes |
# |---|---|---|
# | open-source | CBC (pulp's default), HiGHS, SCIP, GLPK | free; fine for small and medium problems |
# | commercial | Gurobi, CPLEX, FICO Xpress | fastest on hard and large ILO; free academic licenses |
#
# ### Quality
#
# Solvers are compared on public sets of benchmark problems, such as the benchmarks that
# Hans Mittelmann maintains at <https://plato.asu.edu/bench.html>: how many of the
# problems a solver solves within a time limit, and how long it takes on average. On hard
# ILO problems the commercial solvers solve more problems, and solve them faster, than
# the open-source ones. For the models in this course, CBC is more than fast enough. The
# commercial solvers cost money outside academia, which is why not every company uses
# the fastest one.
#
# ### Speedup
#
# For LO, the classic method is the simplex algorithm (corner to corner). Since the 1980s,
# interior-point methods move through the interior of the feasible region instead and solve
# LO in provably polynomial time; modern solvers offer both. For ILO, branch and bound (see
# [Integer Optimization](lecture8_integer-optimization.ipynb#branch-and-bound)) wraps
# around an LO solver, with many improvements on top: preprocessing that simplifies the
# model before the search starts, extra constraints (cuts) that tighten the LO
# relaxations, and heuristics that look for good integer solutions. ILO solver
# performance has improved by a factor of roughly 1000 between 2000 and 2020 through
# better algorithms alone, comparable to the hardware speed-up over the same period.
#
# pulp can call any installed solver without changing the model; only the `.solve(...)`
# line changes. To see what is available here:

# %%
import re

import numpy as np
import plotly.graph_objects as go
import pulp

print(pulp.listSolvers(onlyAvailable=True))

# %% [markdown]
# Without installing anything, you can also export the model to a standard `.mps` file and
# submit it to the free [NEOS Server](https://neos-server.org/neos/), which hosts many
# solvers including commercial ones. If the variable and constraint names might leak
# information about your data, pulp can anonymize them on export with `rename=1`.

# %% [markdown]
# ## Reading the Solve Log
#
# ### An Example That Takes Some Effort
#
# The small examples of the previous lectures are solved before there is anything to
# watch. To see a solver at work, we take the single-machine scheduling model from
# [Machine Scheduling](lecture9_machine-scheduling.ipynb#machine-scheduling) and give it
# 10 jobs instead of 3. The job data (durations, release dates, due dates and tardiness
# costs) is drawn at random with a fixed seed, so every run gets the same instance. The
# model code is the same as in Lecture 9, placed in a function so that we can build it
# again for other instances.


# %%
def random_jobs(
    n_jobs: int, seed: int
) -> tuple[list[str], dict[str, int], dict[str, int], dict[str, int], dict]:
    """Random single-machine scheduling instance with n_jobs jobs."""
    rng = np.random.default_rng(seed)
    jobs = [f"J{k + 1}" for k in range(n_jobs)]
    duration = {i: int(rng.integers(2, 10)) for i in jobs}
    release = {i: int(rng.integers(0, 20)) for i in jobs}
    slack = {i: int(rng.integers(0, 15)) for i in jobs}
    due = {i: release[i] + duration[i] + slack[i] for i in jobs}
    tardiness_cost = {i: int(rng.integers(1, 4)) for i in jobs}
    return jobs, duration, release, due, tardiness_cost


def build_single_machine(
    n_jobs: int, seed: int
) -> tuple[pulp.LpProblem, dict, dict, dict]:
    """Single-machine scheduling ILO of Lecture 9 for a random instance."""
    jobs, duration, release, due, tardiness_cost = random_jobs(n_jobs, seed)
    big_m = max(release.values()) + sum(duration.values())

    single_machine = pulp.LpProblem(
        name="single_machine", sense=pulp.LpMinimize
    )
    start = {
        i: pulp.LpVariable(name=f"x_{i}", lowBound=release[i]) for i in jobs
    }
    before = {
        (i, j): pulp.LpVariable(name=f"y_{i}_{j}", cat="Binary")
        for i in jobs
        for j in jobs
        if i != j
    }
    tardy = {i: pulp.LpVariable(name=f"z_{i}", lowBound=0) for i in jobs}

    single_machine += pulp.lpSum(tardiness_cost[i] * tardy[i] for i in jobs)
    for i in jobs:
        finish = start[i] + duration[i]
        single_machine += tardy[i] >= finish - due[i], f"tardiness_{i}"
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            finish = start[i] + duration[i]
            switch = big_m * (1 - before[i, j])
            name = f"no_overlap_{i}_{j}"
            single_machine += finish <= start[j] + switch, name
            if i < j:
                pair_order = before[i, j] + before[j, i]
                single_machine += pair_order == 1, f"order_{i}_{j}"
    return single_machine, start, before, tardy


# %% [markdown]
# With `msg=True`, CBC writes its log to the place where Python itself runs. In a
# notebook (also in Colab) that is often not the notebook, so the log does not always
# appear below the cell. Passing `logPath` writes the log to a file instead, which we
# can then read and print. The full log is long; the lines that matter most are printed
# below.

# %%
schedule, start, before, tardy = build_single_machine(n_jobs=10, seed=1)
schedule.solve(pulp.PULP_CBC_CMD(msg=False, logPath="cbc.log"))
with open("cbc.log") as log_file:
    schedule_log = log_file.read()

KEY_LINES = ("Cbc0010I", "Cbc0001I", "Result")
for line in schedule_log.splitlines():
    if "Integer solution of" in line or line.startswith(KEY_LINES):
        print(line)
print("status:", pulp.LpStatus[schedule.status])
print("total weighted tardiness:", schedule.objective.value())

# %% [markdown]
# ### Incumbent, Best Bound and Gap
#
# Recall from [branch and bound](lecture8_integer-optimization.ipynb#branch-and-bound)
# that the solver keeps two numbers during the search. For a minimization problem:
#
# - The **incumbent** is the best integer solution found so far. Its objective value is
#   an upper bound on the optimal value: the optimum is at least as good. In the CBC log,
#   every line `Integer solution of ... found` reports a new incumbent, and `best
#   solution` in a progress line is the incumbent's value.
# - The **best bound** is a lower bound on the optimal value: no integer solution can be
#   better. It is the smallest LO relaxation value among the nodes of the tree that are
#   still open, and it can only go up during the search. CBC calls it `best possible`.
#
# The **gap** is the distance between the two, relative to the incumbent:
#
# $$
# \text{gap} = \frac{\text{incumbent} - \text{best bound}}{|\text{incumbent}|}.
# $$
#
# A gap of 0 means that the incumbent is proven optimal, and that is when the solver
# stops. (Solvers differ slightly in what they divide by; CBC's summary line at the end of
# a stopped run divides by the best bound.)
#
# In the log above, CBC finds an incumbent right away, before branching starts, with a
# heuristic called the feasibility pump. At the root node the best bound is about 6,
# while the incumbent is almost 80. The later incumbents are found during the search, and
# the last one is the optimum. After that, CBC keeps working until the bound has come up
# to the incumbent: the line `Search completed` is the moment optimality is proven.
# [](#fig-progress-10) shows both numbers over time.


# %% tags=["hide-input"]
Event = tuple[float, float | None, float | None]
NUMBER = r"(-?[\d.e+]+)"
SECONDS = r".*\(([\d.]+) seconds\)"
# pattern, and which groups hold incumbent, bound and time
Pattern = tuple[str, int, int | None, int]
NEW_INCUMBENT: Pattern = (r"solution of " + NUMBER + SECONDS, 1, None, 2)
# log line code -> pattern and groups, for the other lines we use
LOG_PATTERNS: dict[str, Pattern] = {
    "Cbc0010I": (
        NUMBER + r" best solution, best possible " + NUMBER + SECONDS,
        1,
        2,
        3,
    ),
    "Cbc0001I": (r"best objective " + NUMBER + SECONDS, 1, 1, 2),
    "Cbc0005I": (
        r"best objective " + NUMBER + r" \(best possible " + NUMBER + SECONDS,
        1,
        2,
        3,
    ),
}


def read_progress(log_text: str) -> list[Event]:
    """(seconds, incumbent, best bound) events from a CBC log."""
    events: list[Event] = []
    for line in log_text.splitlines():
        if line.startswith("Cbc0045I"):
            # a warm start: the incumbent at time 0
            cost = re.search(r"cost " + NUMBER, line)
            if cost:
                events.append((0.0, float(cost[1]), None))
            continue
        if "Integer solution of" in line:
            # a new incumbent (the code depends on how it was found)
            pattern, inc_group, bound_group, time_group = NEW_INCUMBENT
        elif line[:8] in LOG_PATTERNS:
            code = line[:8]
            pattern, inc_group, bound_group, time_group = LOG_PATTERNS[code]
        else:
            continue
        found = re.search(pattern, line)
        if not found:
            continue
        incumbent: float | None = float(found[inc_group])
        if incumbent is not None and incumbent > 1e40:
            # CBC's placeholder while there is no incumbent yet
            incumbent = None
        bound = float(found[bound_group]) if bound_group else None
        events.append((float(found[time_group]), incumbent, bound))
    return events


TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.22,
    },
}
AXIS_STYLE = {
    "gridcolor": "rgba(128,128,128,0.25)",
    "zerolinecolor": "rgba(128,128,128,0.5)",
}
# a static picture: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}


def plot_progress(log_text: str) -> go.Figure:
    """Incumbent and best bound over time, as step lines."""
    incumbent_t: list[float] = []
    incumbent_v: list[float] = []
    bound_t: list[float] = []
    bound_v: list[float] = []
    for sec, incumbent, bound in read_progress(log_text):
        if incumbent is not None:
            incumbent_t.append(sec)
            incumbent_v.append(incumbent)
        if bound is not None:
            bound_t.append(sec)
            bound_v.append(bound)
    end = max(incumbent_t + bound_t)
    progress = go.Figure()
    progress.add_trace(
        go.Scatter(
            x=incumbent_t + [end],
            y=incumbent_v + incumbent_v[-1:],
            mode="lines+markers",
            line={"shape": "hv", "color": "#1f77b4", "width": 2},
            name="incumbent (upper bound)",
        )
    )
    progress.add_trace(
        go.Scatter(
            x=bound_t + [end],
            y=bound_v + bound_v[-1:],
            mode="lines+markers",
            line={"shape": "hv", "color": "#ff7f0e", "width": 2},
            name="best bound (lower bound)",
        )
    )
    progress.update_layout(**PLOT_LAYOUT, height=360)
    time_axis = {"title": "time (seconds)", "rangemode": "tozero"}
    value_axis = {"title": "total weighted tardiness", "rangemode": "tozero"}
    progress.update_xaxes(**time_axis, **AXIS_STYLE)
    progress.update_yaxes(**value_axis, **AXIS_STYLE)
    return progress


# %% label="progress-10" tags=["remove-cell"]
plot_progress(schedule_log).show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #progress-10
# :label: fig-progress-10
#
# Incumbent and best bound during the solve of the 10-job scheduling instance. CBC only
# reports the best bound at the root node, every 1000 nodes, and at the end, so in
# reality it rises in more steps than the plot shows.
# :::

# %% [markdown]
# ### Finding Versus Proving
#
# A log like this one tells you which part of the work is hard for the solver. Three
# patterns come back again and again.
#
# The first pattern is the one above, and it is the most common one for ILO models: the
# optimum (or a solution close to it) is found early, and most of the time goes into
# proving that nothing better exists. The reason is a weak best bound. In this model the
# LO relaxation may set each $y_{ij}$ to $\tfrac12$, so every no-overlap constraint is
# relaxed by $M/2$ and the jobs may overlap. We can check this by solving only the LO
# relaxation, which `mip=False` does:

# %%
schedule.solve(pulp.PULP_CBC_CMD(msg=False, mip=False))
print("LO relaxation value:", schedule.objective.value())

# %% [markdown]
# The LO relaxation allows zero tardiness, while the optimum has a total weighted
# tardiness of 75. Branch and bound has to close this whole gap by branching.
#
# The second pattern appears when an instance is too large to solve to optimality in the
# time we have. Then the incumbent improves step by step over time, and the question is
# when it is good enough to stop. With 12 jobs, CBC does not finish within 30 seconds. We
# stop it after 5 seconds with a time limit, `timeLimit=5`:

# %%
schedule12, start12, before12, tardy12 = build_single_machine(
    n_jobs=12, seed=1
)
schedule12.solve(pulp.PULP_CBC_CMD(msg=False, logPath="cbc.log", timeLimit=5))
with open("cbc.log") as log_file:
    schedule12_log = log_file.read()

for line in schedule12_log.splitlines():
    if line.startswith(("Cbc0005I", "Result", "Objective", "Lower bound")):
        print(line)
print("status:", pulp.LpStatus[schedule12.status])
print("solution status:", pulp.LpSolution[schedule12.sol_status])

# %% [markdown]
# Two things stand out. First, `pulp.LpStatus` says `Optimal`, even though CBC stopped on
# the time limit. The solution is feasible, but not proven optimal: pulp's
# `sol_status` tells the difference (`Solution Found` rather than `Optimal Solution
# Found`), and the log says `Stopped on time limit`. Always check one of these after a
# solve with a limit. Second, the best bound has hardly moved from its root value, so the
# gap is still very large. The incumbent may well be close to optimal, but the solver
# cannot show it ([](#fig-progress-12)).

# %% label="progress-12" tags=["remove-cell"]
plot_progress(schedule12_log).show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #progress-12
# :label: fig-progress-12
#
# Incumbent and best bound for the 12-job instance, stopped after 5 seconds. The
# incumbent improves quickly at first and then more slowly; the best bound stays far
# below it.
# :::

# %% [markdown]
# The third pattern is a solver that has a hard time finding any integer solution at
# all: the log shows many nodes but no `Integer solution` line, and the incumbent in
# the progress lines stays at a huge placeholder value (CBC prints `1e+50`). Without an
# incumbent, the solver can prune nothing by bound, and a time limit leaves you with
# nothing. This happens with many equality constraints or tight hard deadlines, where
# few combinations of integer values are feasible. The remedy is to give the solver a
# feasible solution to start from, which is the warm start below.

# %% [markdown]
# ## Helping a Solver
#
# Once the log shows where the difficulty is, there are three things you can do about
# it.
#
# ### Accept a Good Solution: Time Limits and Gap Tolerances
#
# Data such as forecast demand or processing times is rarely known up to 1%, so a
# solution that is proven to be within 1% of optimal is usually good enough. A **gap
# tolerance** tells the solver to stop as soon as the gap is below a given fraction:
# `gapRel=0.01` in `PULP_CBC_CMD` stops at a gap of 1%. A time limit (`timeLimit`, in
# seconds) stops the solver after a fixed time, whatever the gap. Both are
# options in the same call:
#
# ```python
# schedule.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0.01, timeLimit=60))
# ```
#
# A gap tolerance only helps if the best bound comes close to the incumbent before the
# end. In the scheduling examples it does not: the bound stays low until the very end,
# so `gapRel=0.01` would not stop the search any earlier, and only a time limit saves
# time. The price is that you then do not know how good the solution is.
#
# ### Warm Starting
#
# In a **warm start**, you give the solver a feasible solution to begin with. It then
# has an incumbent from the start: this helps with the third pattern above, and it lets
# the solver prune nodes by bound from the first node on. Good starting solutions come
# from a simple rule of thumb (a heuristic, see
# [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb)), or from the
# solution of an earlier, almost identical problem, such as yesterday's schedule when
# one job has changed.
#
# For the scheduling instance, a simple rule is earliest due date first: process the
# jobs in order of their due dates, each as soon as the machine is free and the job is
# released. In pulp, `.setInitialValue(...)` sets a variable's starting value, and
# `warmStart=True` passes these values to CBC. The starting solution must give a value to
# every variable, including the binary order variables.

# %%
schedule, start, before, tardy = build_single_machine(n_jobs=10, seed=1)
jobs, duration, release, due, tardiness_cost = random_jobs(n_jobs=10, seed=1)

machine_free = 0
edd_start: dict[str, int] = {}
for i in sorted(jobs, key=lambda job: due[job]):
    edd_start[i] = max(machine_free, release[i])
    machine_free = edd_start[i] + duration[i]

for i in jobs:
    start[i].setInitialValue(edd_start[i])
    lateness = edd_start[i] + duration[i] - due[i]
    tardy[i].setInitialValue(max(0, lateness))
for (i, j), order_var in before.items():
    order_var.setInitialValue(1 if edd_start[i] < edd_start[j] else 0)

schedule.solve(pulp.PULP_CBC_CMD(msg=False, logPath="cbc.log", warmStart=True))
with open("cbc.log") as log_file:
    warm_log = log_file.read()

WARM_LINES = ("Cbc0045I", "Cbc0001I")
for line in warm_log.splitlines():
    if "Integer solution of" in line or line.startswith(WARM_LINES):
        print(line)
print("total weighted tardiness:", schedule.objective.value())

# %% [markdown]
# The line `Cbc0045I MIPStart provided solution` shows that CBC accepted the starting
# solution as its first incumbent. The search then finds the same optimum as before,
# in about the same time: a warm start helps to find good solutions, not to prove
# optimality, and here finding was never the hard part. Its value shows in problems of
# the third pattern, and in loops that re-solve a large ILO many times with small data
# changes, such as the simulation-optimization loop in
# [Simulation Optimization](lecture13_simulation-optimization.ipynb).
#
# :::{note} Warm starting on Windows
# On your own Windows computer, CBC can only read the starting solution if pulp keeps its
# temporary files: use `pulp.PULP_CBC_CMD(msg=False, warmStart=True, keepFiles=True)`.
# This is not needed in Colab.
# :::
#
# ### A Tighter Formulation
#
# The most effective help is often a better model. The weak bound of the scheduling
# model comes from its big-M constraints, and a large $M$ makes the LO relaxation weaker.
# That is why [Machine Scheduling](lecture9_machine-scheduling.ipynb#big-m-indicator)
# chooses $M$ as small as possible while keeping the model correct. For this instance
# even the smallest valid $M$ leaves an LO relaxation value of 0, and other formulations
# of the same problem, with more variables, give much stronger bounds. Choosing between
# formulations goes beyond this course, but the log tells you when it is worth trying:
# when the best bound hardly moves.
#
# :::{exercise}
# :label: ex-10-1
#
# Solve the scheduling instance with `build_single_machine` for 8, 10 and 12 jobs (seed
# 1), with a time limit of 20 seconds. For each size, read from the log when the last
# incumbent was found and whether optimality was proven. How does the share of the time
# spent on proving change with the number of jobs?
# :::
#
# :::{exercise}
# :label: ex-10-2
#
# Warm start [the shift-scheduling example](lecture9_covering.ipynb#shift-scheduling) from
# the previous day's optimal schedule when one interval's demand changes slightly. Compare
# the number of explored nodes (`Enumerated nodes` in the log) with and without the warm
# start.
# :::

# %% [markdown]
# ## GenAI and Optimization
#
# Large language models (LLMs), the models behind GenAI assistants such as ChatGPT or
# Claude, are studied for optimization in roughly three ways.
#
# The first is to let the LLM generate solutions directly. Yang et al. (2024) describe the
# problem and a few earlier solutions with their objective values in a prompt, ask the
# LLM for a better solution, and repeat. On small traveling salesman problems this finds
# solutions as good as some simple heuristics, but the authors do not aim to beat
# dedicated solvers, and larger instances no longer fit in a prompt. Their main
# application is a different one: improving the prompts themselves. Keep in mind what the
# first part of this notebook showed: a solver gives an incumbent and a bound, so it can
# tell you how good its solution is. An LLM only gives an incumbent, without any
# guarantee.
#
# The second is to make the LLM part of an optimization method. Liu et al. (2024) use an
# LLM inside an evolutionary algorithm (a heuristic from
# [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb)), where the LLM
# combines and changes solutions to create new ones. In adaptive large neighborhood
# search (ALNS), a heuristic that repeatedly destroys part of a solution and repairs it,
# an LLM can write the code of the repair step. Brouwer (2025) compares letting an LLM
# write repair steps in one go with improving them over several rounds in an
# evolutionary way, on a flow shop scheduling problem and a variant of the traveling
# salesman problem; the evolutionary approach works best.
#
# The third is to let the LLM assist the people involved. OptiMUS
# (AhmadiTeshnizi et al., 2024) assists the modeler: from a problem description in plain
# language, it writes an LO or ILO model and the solver code, runs and debugs the code,
# checks the solution, and improves the model and code where needed. Wasserkrug et al.
# (2024) aim at the decision-maker, who usually has no optimization expert at hand. They
# propose research towards a Decision Optimization CoPilot: an assistant that talks with
# the decision-maker in plain language to understand the business problem, and then
# formulates and solves the optimization model. Their experiments with ChatGPT show that
# LLMs can already do part of this, but that much research is still needed.
#
# The third use is the one you will meet most yourself, when you ask an assistant to help
# with a pulp model. It can be a useful pair-programmer if used carefully:
#
# - paste the actual code and the actual error or solver log, not a paraphrase;
# - always re-check a suggested model against your original formulation: it can flip a
#   constraint's direction, quietly redefine a variable, or shift an index range, and pulp
#   will build and solve the wrong model without complaint;
# - `print(problem)` before and after any suggested change shows exactly what moved;
# - for an `"Infeasible"` or `"Unbounded"` status, give it the full model text and ask it to
#   find the conflicting or missing constraint.

# %% [markdown]
# (modeling-tools)=
# ## A Note on Modeling Tools
#
# Before Python libraries like pulp, models were usually written in an algebraic modeling
# language (AML), such as AMPL, GAMS or AIMMS: a language for writing a model in
# near-mathematical notation, kept separate from the data, and handed to whichever solver
# you choose. pulp gives the same model and data separation
# ([Separating Data from the Model](lecture8_linear-optimization.ipynb#separating-data)) inside Python,
# together with everything Python offers for preparing data and analyzing solutions.
# AMLs are still in use, mostly for large models that have been maintained for years, but
# for new projects a Python library is now the usual choice.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.6 "Modeling Tools".
#   AMPL and spreadsheet material replaced with pulp.
# - AhmadiTeshnizi, A., Gao, W., & Udell, M. (2024). OptiMUS: Scalable optimization
#   modeling with (MI)LP solvers and large language models. arXiv:2402.10172.
#   https://arxiv.org/abs/2402.10172
# - Brouwer, V. (2025). *AI-Driven Optimization in ALNS: Generating Repair Operators*.
#   Master's thesis, Information Studies (Data Science), University of Amsterdam.
#   https://dspace.uba.uva.nl/server/api/core/bitstreams/acb54db9-f55c-4da9-9158-92a1d11ad50f/content
# - Liu, S., Chen, C., Qu, X., Tang, K., & Ong, Y.-S. (2024). Large language models as
#   evolutionary optimizers. *2024 IEEE Congress on Evolutionary Computation (CEC)*.
#   https://ieeexplore.ieee.org/document/10611913
# - Mittelmann, H. Benchmarks for optimization software. https://plato.asu.edu/bench.html
# - Wasserkrug, S., Boussioux, L., den Hertog, D., Mirzazadeh, F., Birbil, Ş. İ., Kurtz,
#   J., & Maragno, D. (2024). From large language models and optimization to decision
#   optimization CoPilot: A research manifesto. arXiv:2402.16269.
#   https://arxiv.org/abs/2402.16269
# - Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., & Chen, X. (2024). Large
#   language models as optimizers. *International Conference on Learning Representations
#   (ICLR)*. https://arxiv.org/abs/2309.03409
# - PuLP documentation: https://coin-or.github.io/pulp/
