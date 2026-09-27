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
# description: "Single-machine scheduling as an ILO model, and the modeling tricks it uses: binary variables with a big M for either/or choices, fixed costs and conditional constraints."
# thumbnail: null
# ---
# # Lecture 9: Machine Scheduling and Modeling Tricks
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_machine-scheduling.ipynb)

# %% [markdown]
# The previous notebooks showed problems that are ILO fairly directly. Single-machine
# scheduling is different: two of its requirements, that jobs do not overlap and that
# lateness is only penalized when a job is actually late, are not linear at first
# sight. We model the problem in the same steps as the other applications in this
# lecture, and introduce the *tricks* that make it linear where the model needs them.
# After the model, we look at these tricks in general: binary variables combined with a
# big M turn either/or choices, fixed costs and conditional constraints into linear
# constraints.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate a single-machine scheduling problem as an ILO model;
# - formulate either/or (disjunctive) constraints as linear constraints with a binary
#   variable and a big M;
# - model fixed costs and conditional constraints with binary indicator variables.

# %%
import pulp

# %% [markdown]
# (machine-scheduling)=
# ## Single-Machine Scheduling
#
# ### Practical Motivation
#
# A production facility receives a list of production orders (jobs), which it processes
# one at a time on a single machine. Some orders only become available later (the
# material has not arrived yet), and every order has a promised delivery date. Finishing
# late costs money, for example a contractual penalty per day, and some customers are
# more important than others. In which order, and when, should the jobs be processed so
# that the total penalty is as low as possible? Related problems are project planning
# and planning in healthcare, such as operating-room scheduling.

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# There are $n$ jobs. Job $i$ has a *duration* $s_i$, a *release date* $r_i$ before which
# it cannot start, a *due date* $d_i$, and a *tardiness cost* $c_i$ per time unit it
# finishes after its due date. The machine processes one job at a time, and a job that
# has started runs until it is finished. The problem is to find a schedule (a start time
# per job) that minimizes the total weighted tardiness, where the tardiness of a job is
# how long it finishes after its due date (0 if it is on time).
#
# As an example, take $n = 3$ jobs:
#
# | job $i$ | duration $s_i$ | release date $r_i$ | due date $d_i$ | tardiness cost $c_i$ |
# |---|---|---|---|---|
# | 1 | 6 | 0 | 8 | 1 |
# | 2 | 4 | 2 | 7 | 1 |
# | 3 | 5 | 1 | 12 | 1 |
#
# For example, processing the jobs in the order 1, 3, 2 without idle time gives job 1
# the time interval 0-6, job 3 the interval 6-11 and job 2 the interval 11-15. Jobs 1
# and 3 are on time, and job 2 finishes $15 - 7 = 8$ time units late, so this schedule
# has a total weighted tardiness of 8.
#
# #### Decision Variables
#
# A schedule is fully described by when each job starts, so the natural decision
# variables are
#
# $$
# x_i = \text{start time of job } i, \quad i = 1, \dots, n.
# $$
#
# The finish time of job $i$ is then $x_i + s_i$. We will see below that two more sets
# of variables are needed to write the objective and the constraints linearly: a binary
# $y_{ij}$ for the order of each pair of jobs, and a $z_i$ for the tardiness of each job.
#
# #### Objective
#
# The tardiness of job $i$ is $\max(x_i + s_i - d_i, 0)$, so we want to minimize
#
# $$
# \sum_{i=1}^{n} c_i \max(x_i + s_i - d_i, 0).
# $$
#
# The maximum is not linear. But since we minimize it, the trick from the
# [project-planning](lecture8_linear-optimization.ipynb#project-planning) model works:
# introduce a variable $z_i$ for the tardiness of job $i$ that must be at least both
# terms of the maximum,
#
# $$
# z_i \ge x_i + s_i - d_i, \quad z_i \ge 0, \quad i = 1, \dots, n,
# $$
#
# and minimize $\sum_{i=1}^{n} c_i z_i$ instead. These constraints only say that $z_i$
# is at least the tardiness, but because $c_i > 0$, the optimizer pushes every $z_i$
# down until one of them is met exactly, so that $z_i = \max(x_i + s_i - d_i, 0)$ at the
# optimum.
#
# #### Constraints
#
# **Release dates.** A job cannot start before it is released:
#
# $$
# x_i \ge r_i, \quad i = 1, \dots, n.
# $$
#
# **No overlap.** If job $i$ is processed before job $j$, then job $j$ can only start
# once job $i$ has finished: $x_i + s_i \le x_j$. If $j$ comes first, we need
# $x_j + s_j \le x_i$ instead. Exactly one of the two must hold, but which one is not
# known in advance: the order is itself something to decide. So we add a binary decision
# variable for the order of each pair of jobs,
#
# $$
# y_{ij} =
# \begin{cases}
# 1 & \text{if job } i \text{ is processed before job } j, \\
# 0 & \text{otherwise,}
# \end{cases}
# \quad i \ne j,
# $$
#
# with $y_{ij} + y_{ji} = 1$, since one of the two jobs comes first. We then switch the
# constraint $x_i + s_i \le x_j$ on or off with a *big M*, a constant larger than any
# difference in time that can occur in a sensible schedule:
#
# $$
# x_i + s_i \le x_j + M(1 - y_{ij}), \quad i \ne j.
# $$
#
# For a pair of jobs $i$ and $j$ this gives two constraints, one for $y_{ij}$ and one for
# $y_{ji}$, and the order decides which one does the work:
#
# | order | $y_{ij}$ | $y_{ji}$ | $x_i + s_i \le x_j + M(1 - y_{ij})$ | $x_j + s_j \le x_i + M(1 - y_{ji})$ |
# |---|---|---|---|---|
# | $i$ before $j$ | 1 | 0 | $x_i + s_i \le x_j$ | $x_j + s_j \le x_i + M$ (no restriction) |
# | $j$ before $i$ | 0 | 1 | $x_i + s_i \le x_j + M$ (no restriction) | $x_j + s_j \le x_i$ |
#
# $M = \max_i r_i + \sum_i s_i$ is large enough: all jobs can be finished by then, so no
# sensible schedule has a job finishing more than $M$ after another job starts.
#
# #### Complete ILO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} c_i z_i & \text{(weighted tardiness)} \\
# \text{s.t.} \quad & x_i \ge r_i, \quad i = 1, \dots, n & \text{(release dates)} \\
# & x_i + s_i \le x_j + M(1 - y_{ij}), \quad i \ne j & \text{(no overlap)} \\
# & y_{ij} + y_{ji} = 1, \quad i < j & \text{(order)} \\
# & z_i \ge x_i + s_i - d_i, \quad i = 1, \dots, n & \text{(tardiness)} \\
# & y_{ij} \in \{0, 1\}, \quad i \ne j \\
# & z_i \ge 0, \quad i = 1, \dots, n.
# \end{aligned}
# $$
#
# The model grows fast: for $n$ jobs it has $n^2 + n$ variables and
# $1.5n^2 + 0.5n$ constraints (not counting $z_i \ge 0$ and the binary constraints),
# mostly because of the $y_{ij}$.

# %% [markdown]
# ### Modeling the Example
#
# For the three jobs, $M = \max_i r_i + \sum_i s_i = 2 + 15 = 17$, and the model is
#
# $$
# \begin{aligned}
# \min \quad & z_1 + z_2 + z_3 \\
# \text{s.t.} \quad & x_1 \ge 0 & \text{(release dates)} \\
# & x_2 \ge 2 \\
# & x_3 \ge 1 \\
# & x_1 + 6 \le x_2 + 17(1 - y_{12}) & \text{(no overlap)} \\
# & x_1 + 6 \le x_3 + 17(1 - y_{13}) \\
# & x_2 + 4 \le x_1 + 17(1 - y_{21}) \\
# & x_2 + 4 \le x_3 + 17(1 - y_{23}) \\
# & x_3 + 5 \le x_1 + 17(1 - y_{31}) \\
# & x_3 + 5 \le x_2 + 17(1 - y_{32}) \\
# & y_{12} + y_{21} = 1 & \text{(order)} \\
# & y_{13} + y_{31} = 1 \\
# & y_{23} + y_{32} = 1 \\
# & z_1 \ge x_1 + 6 - 8 & \text{(tardiness)} \\
# & z_2 \ge x_2 + 4 - 7 \\
# & z_3 \ge x_3 + 5 - 12 \\
# & y_{ij} \in \{0, 1\}, \quad i, j = 1, 2, 3,\ i \ne j \\
# & z_1, z_2, z_3 \ge 0.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# Three things in the code below are new. The release dates are not added as separate
# constraints but as lower bounds of the start-time variables (`lowBound=release[i]`),
# which means the same to the solver. The order variables are binary
# (`cat="Binary"`). And the big M is computed from the data instead of typed in, so it
# stays large enough when the data changes.

# %%
jobs = [1, 2, 3]
duration = {1: 6, 2: 4, 3: 5}
release = {1: 0, 2: 2, 3: 1}
due = {1: 8, 2: 7, 3: 12}
tardiness_cost = {1: 1, 2: 1, 3: 1}
big_m = max(release.values()) + sum(duration.values())

single_machine = pulp.LpProblem(name="single_machine", sense=pulp.LpMinimize)
start = {i: pulp.LpVariable(name=f"x_{i}", lowBound=release[i]) for i in jobs}
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
        single_machine += finish <= start[j] + switch, f"no_overlap_{i}_{j}"
        if i < j:
            pair_order = before[i, j] + before[j, i]
            single_machine += pair_order == 1, f"order_{i}_{j}"

single_machine.solve(pulp.PULP_CBC_CMD(msg=False))
order = sorted(jobs, key=lambda i: start[i].value())
print("status:", pulp.LpStatus[single_machine.status])
print("start times:", {i: start[i].value() for i in jobs})
print("processing order:", order)
print("total weighted tardiness:", single_machine.objective.value())

# %% [markdown]
# The optimal order is 1, 2, 3, with a total weighted tardiness of 6: jobs 2 and 3 are
# each 3 time units late. That is better than the order 1, 3, 2 from the example above,
# which had tardiness 8.

# %% [markdown]
# ### Extensions
#
# #### Other Objectives
#
# The same variables support other common objectives just by changing the line we
# minimize:
#
# - **flowtime** $\sum_i (x_i + s_i - r_i)$: total time jobs spend in the system;
# - **makespan** $\max_i (x_i + s_i)$: when the machine finishes everything; minimize it
#   with the project-planning trick, $\min z$ subject to $x_i + s_i \le z$ for all $i$;
# - a **weighted sum** of makespan and tardiness (or: minimize tardiness first, then
#   minimize makespan with the tardiness fixed at its optimum).

# %% [markdown]
# ## Modeling Tricks Used in This Model
#
# The scheduling model used a binary variable and a big M to switch a constraint on or
# off. The same idea shows up in many other models. This section describes it in
# general.
#
# (big-m-indicator)=
# ### Big M and Indicator Variables
#
# Suppose a cost is incurred only when an activity is *used* at all, regardless of how
# much. Take the [transportation problem](lecture9_transportation.ipynb) with an extra
# fixed cost $K$ for every link that carries any flow. Introduce a binary $y_{ij}$
# meaning "link $i \to j$ is used", add $K \sum_{i,j} y_{ij}$ to the objective, and tie
# $y_{ij}$ to $x_{ij}$ with a big M, a constant larger than any $x_{ij}$ could ever be:
#
# $$
# x_{ij} \le M\, y_{ij}.
# $$
#
# If $x_{ij} > 0$ then $y_{ij}$ is forced to 1. If $x_{ij} = 0$ then $y_{ij}$ could be 0
# or 1, but since $K > 0$ and we are minimizing, the optimizer sets it to 0. So
# effectively $y_{ij} = 1 \Leftrightarrow x_{ij} > 0$. A binary variable used this way,
# to indicate whether something happens, is called an **indicator variable**.
#
# ### Conditional Constraints
#
# The same idea handles a constraint that should hold only under a condition. If binary
# $y$ represents the condition and the constraint is $x \le b$, then
#
# $$
# x \le b + (1 - y) M
# $$
#
# is the original constraint when $y = 1$, and no constraint at all (right-hand side
# huge) when $y = 0$. The no-overlap constraints of the scheduling model are of this
# form, with the condition "job $i$ comes before job $j$".
#
# ### Either/Or (Disjunctive) Constraints
#
# In the scheduling model, for every pair of jobs one of two constraints had to hold,
# and we did not know in advance which one. In general, for two constraints
# $f(x) \le 0$ and $g(x) \le 0$ of which at least one must hold, introduce a binary $y$
# and a big M:
#
# $$
# \begin{aligned}
# f(x) &\le (1 - y) M \\
# g(x) &\le y M \\
# y &\in \{0, 1\}.
# \end{aligned}
# $$
#
# When $y = 1$ the first constraint is enforced and the second is switched off; when
# $y = 0$ the reverse. In the scheduling model we used two binaries $y_{ij}$ and $y_{ji}$
# with $y_{ij} + y_{ji} = 1$ instead of a single $y$, which is the same thing written
# symmetrically.
#
# Choose $M$ as small as possible while still switching the constraint off: a very large
# $M$ makes the LO relaxations that [branch and
# bound](lecture8_integer-optimization.ipynb) solves weak, and can cause numerical
# trouble in the solver.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.7 "Modeling Tricks."
#   The book's AMPL example is replaced with pulp.
