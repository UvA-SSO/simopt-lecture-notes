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
# # Lecture 9: Machine Scheduling and Modeling Tricks

# %% [markdown]
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_advanced-modeling.ipynb)

# %% [markdown]
# The previous notebook showed problems that are ILO fairly directly. This one is about the
# *tricks* that turn an apparently nonlinear requirement (an either/or choice, a fixed cost
# that only applies when something is used, a constraint that only holds under a condition)
# into linear constraints with binary variables. We build up to single-machine scheduling,
# which uses several of these tricks at once.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - model fixed costs and either/or conditions with big-M constraints and binary indicator
#   variables;
# - formulate disjunctive constraints as linear constraints;
# - formulate a single-machine scheduling problem as an ILO model.

# %%
try:
    import pulp
except ModuleNotFoundError:  # pulp is not preinstalled on Google Colab
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pulp"], check=True)
    import pulp

# %% [markdown]
# ## Big M and Indicator Variables
#
# Suppose a cost is incurred only when an activity is *used* at all, regardless of how much.
# Take the transportation problem from
# [Applications of (I)LO](lecture9_ilo-applications.ipynb) with an extra fixed cost $K$ for
# every link that carries any flow. Introduce a binary $y_{ij}$ meaning "link $i \to j$ is
# used", add $K \sum_{i,j} y_{ij}$ to the objective, and tie $y_{ij}$ to $x_{ij}$ with a
# big M: a constant larger than any $x_{ij}$ could ever be:
#
# $$
# x_{ij} \le M\, y_{ij}.
# $$
#
# If $x_{ij} > 0$ then $y_{ij}$ is forced to 1. If $x_{ij} = 0$ then $y_{ij}$ could be 0 or
# 1, but since $K > 0$ and we are minimizing, the optimizer sets it to 0. So effectively
# $y_{ij} = 1 \Leftrightarrow x_{ij} > 0$.
#
# The same idea handles a constraint that should hold only under a condition. If binary
# $y$ represents the condition and the constraint is $x \le b$, then
#
# $$
# x \le b + (1 - y) M
# $$
#
# is the original constraint when $y = 1$, and no constraint at all (right-hand side huge)
# when $y = 0$.
#
# :::{exercise}
# :label: ex-6-18
#
# Take [the transportation exercise](lecture9_ilo-applications.ipynb#ex-6-8) and add a fixed
# cost of 10 for every link that is used. Solve the resulting ILO with pulp.
# :::

# %% [markdown]
# ## Either/Or (Disjunctive) Constraints
#
# Two jobs, A and B, must run one after the other on the same machine, but *which* goes
# first is a decision. Let $x_A, x_B$ be start times and $d_A, d_B$ durations. If A is first
# we need $x_A + d_A \le x_B$; if B is first we need $x_B + d_B \le x_A$. Exactly one of
# these must hold. With a binary $y$ ($y = 1$ meaning "A before B") and a big $M$:
#
# $$
# \begin{aligned}
# x_A + d_A &\le x_B + (1 - y) M, \\
# x_B + d_B &\le x_A + y M, \\
# y &\in \{0, 1\}.
# \end{aligned}
# $$
#
# When $y = 1$ the first constraint is active and the second is slack; when $y = 0$ the
# reverse. This "disjunctive" pattern is the heart of the scheduling model below.

# %% [markdown]
# (machine-scheduling)=
# ## Single-Machine Scheduling
#
# A production facility processes a list of jobs one at a time. Job $i$ has a *duration*
# $s_i$, a *release date* $r_i$ before which it cannot start, a *due date* $d_i$, and a
# *tardiness cost* $c_i$ per time unit it finishes late. We want a schedule (a start time
# per job) that minimizes total weighted tardiness. Related problems: project planning,
# operating-room planning.
#
# **Decision variables.** $x_i$ = start time of job $i$; binary $y_{ij}$ = 1 iff job $i$
# runs before job $j$ (for $i \ne j$); $z_i$ = tardiness of job $i$.
#
# **No overlap.** Apply the disjunctive trick to every pair: for all $i \ne j$,
#
# $$
# x_i + s_i \le x_j + M(1 - y_{ij}), \qquad y_{ij} + y_{ji} = 1.
# $$
#
# **Tardiness.** The true tardiness $\max(x_i + s_i - d_i, 0)$ is nonlinear, but since we
# *minimize* it, the same "introduce a variable that is forced up to the max" trick from the
# [project-planning](lecture8_linear-optimization.ipynb#project-planning) model works:
#
# $$
# z_i \ge x_i + s_i - d_i, \qquad z_i \ge 0.
# $$
#
# **Full model.**
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_i c_i z_i \\
# \text{subject to} \quad & x_i \ge r_i \text{ for all } i \\
# & x_i + s_i \le x_j + M(1 - y_{ij}) \text{ for all } i \ne j \\
# & y_{ij} + y_{ji} = 1 \text{ for all } i < j \\
# & z_i \ge x_i + s_i - d_i, \quad z_i \ge 0 \text{ for all } i \\
# & y_{ij} \in \{0, 1\}.
# \end{aligned}
# $$
#
# The model grows fast: about $n^2$ variables and $2n^2 + n$ constraints for $n$ jobs.
#
# Solving a three-job instance:

# %%
jobs = [1, 2, 3]
s = {1: 6, 2: 4, 3: 5}  # duration
r = {1: 0, 2: 2, 3: 1}  # release date
due = {1: 8, 2: 7, 3: 12}  # due date
c = {1: 1, 2: 1, 3: 1}  # tardiness cost
big_m = sum(s.values()) + max(r.values())

sched = pulp.LpProblem(name="single_machine", sense=pulp.LpMinimize)
start = {i: pulp.LpVariable(name=f"x_{i}", lowBound=r[i]) for i in jobs}
before = {
    (i, j): pulp.LpVariable(name=f"y_{i}_{j}", cat="Binary") for i in jobs for j in jobs if i != j
}
tardy = {i: pulp.LpVariable(name=f"z_{i}", lowBound=0) for i in jobs}

sched += pulp.lpSum(c[i] * tardy[i] for i in jobs)
for i in jobs:
    sched += tardy[i] >= start[i] + s[i] - due[i]
for i in jobs:
    for j in jobs:
        if i == j:
            continue
        sched += start[i] + s[i] <= start[j] + big_m * (1 - before[i, j])
        if i < j:
            sched += before[i, j] + before[j, i] == 1

sched.solve(pulp.PULP_CBC_CMD(msg=False))
order = sorted(jobs, key=lambda i: start[i].value())
print("start times:", {i: start[i].value() for i in jobs})
print("processing order:", order, "| total weighted tardiness:", sched.objective.value())

# %% [markdown]
# ### Other Objectives
#
# The same variables support other common objectives just by changing the line we minimize:
#
# - **flowtime** $\sum_i (x_i + s_i - r_i)$: total time jobs spend in the system;
# - **makespan** $\max_i (x_i + s_i)$: when the machine finishes everything; minimize it
#   with the project-planning trick, $\min z$ subject to $x_i + s_i \le z$ for all $i$;
# - a **weighted sum** of makespan and tardiness (or: minimize tardiness first, then
#   minimize makespan with the tardiness fixed at its optimum).
#
# :::{exercise}
# :label: ex-6-20
#
# Assume activities B and C of the
# [project-planning problem](lecture8_linear-optimization.ipynb#project-planning) use the
# same resource and so cannot run at the same time. Formulate this as an ILO and solve it
# with pulp.
# :::
#
# :::{exercise}
# :label: ex-6-21
#
# Implement single-machine scheduling with the total-tardiness objective in pulp and solve
# it for:
#
# | | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
# |---|---|---|---|---|---|---|---|---|---|---|
# | duration | 4 | 5 | 3 | 5 | 7 | 1 | 0 | 3 | 2 | 10 |
# | release time | 3 | 4 | 7 | 11 | 10 | 0 | 0 | 10 | 0 | 15 |
# | due date | 11 | 12 | 20 | 25 | 20 | 10 | 30 | 30 | 10 | 20 |
#
# For a constraint over all $i \ne j$, loop both indices and `continue` when `i == j`.
# Nested binary variables:
# `y = [[pulp.LpVariable(name=f"y_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]`.
# :::
#
# :::{exercise}
# :label: ex-6-22
#
# Change the objective of
# [the class-scheduling exercise](lecture9_ilo-applications.ipynb#ex-6-14) to: minimize the
# time the last class finishes.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.7 "Modeling Tricks."
#   The book's AMPL example is replaced with pulp.
