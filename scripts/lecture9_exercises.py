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
# # Lecture 9: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in
# [Transportation and Transshipment](lecture9_transportation.ipynb),
# [Set Covering and Shift Scheduling](lecture9_covering.ipynb), and
# [Machine Scheduling](lecture9_machine-scheduling.ipynb) check what you just read. This
# notebook collects the larger exercises for Lecture 9: independent problems worth more
# time.

# %% [markdown]
# :::{exercise}
# :label: ex-6-8
#
# Solve the following transportation problem in pulp, where "x" means there is no
# connection (use a very large cost, e.g. `1e6`, in place of $\infty$): source 1 has supply
# 10 and costs $(0, 5, \text{x}, 0)$ to destinations 1–4; source 2 has supply 6 and costs
# $(4, 6, 4, 3)$; source 3 has supply 10 and costs $(2, 4, 4, 6)$; each destination has
# demand 5.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-12
#
# Solve the following ILO problem, inspired by Guenin, Könemann, and Tunçel (2014). A
# swimming pool is open for 12 hours and the lifeguards on duty must be selected. Each
# lifeguard has fixed working hours (first and last hour worked) and a wage. Select the
# cheapest set of lifeguards with at least one on duty every hour. (Ben + Celia + Fred is a
# feasible solution.)
#
# | Lifeguard | Ann | Ben | Celia | Dick | Estelle | Fred |
# |---|---|---|---|---|---|---|
# | Hours | 1–6 | 1–4 | 5–8 | 7–10 | 7–12 | 9–12 |
# | Wage | 8 | 6 | 6 | 3 | 7 | 3 |
#
# Use a matrix (nested list or dict) for $a_{ui}$.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-13
#
# The required staffing in a call center, 9am–9pm in 30-minute intervals, is:
#
# 10, 11, 13, 16, 16, 13, 11, 10, 10, 11, 12, 13, 14, 14, 13, 11, 10, 9, 9, 10, 9, 8, 8, 8.
#
# Two shift types are available:
#
# - 8 hours working, with a 30-minute unpaid break in the middle, wage €20/hr, starting
#   every half hour from 9:00 to 12:30;
# - 4 consecutive hours, wage €24/hr, starting every half hour from 9:00 to 17:00.
#
# Formulate this as a covering problem and solve it with pulp.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-14
#
# For one school day, classes must be assigned to professors so that each class has one hour
# with each professor it needs and no professor teaches two classes at once. A matrix
# $a_{cp}$ gives which classes need which professors ($a_{cp} = 1$ if class $c$ needs
# professor $p$). Formulate an ILO model that minimizes the total number of hours classes
# spend at school (a class stays until right after its last professor hour).
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-15
#
# Pairs must be made among $n$ students; each student lists the students they are willing to
# work with. Formulate an ILO model maximizing the number of pairs, each student in at most
# one pair (not everyone need be paired).
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-18
#
# Take [the transportation exercise](#ex-6-8) and add a fixed cost of 10 for every link that
# is used. Solve the resulting ILO with pulp.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-20
#
# Assume activities B and C of the
# [project-planning problem](lecture8_linear-optimization.ipynb#project-planning) use the
# same resource and so cannot run at the same time. Formulate this as an ILO and solve it
# with pulp.
# :::

# %% [markdown]
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

# %% [markdown]
# :::{exercise}
# :label: ex-6-22
#
# Change the objective of [the class-scheduling exercise](#ex-6-14) to: minimize the time
# the last class finishes.
# :::

# %% [markdown]
# ## References
#
# - Guenin, B., Könemann, J., & Tunçel, L. (2014). *A Gentle Introduction to Optimization*.
#   Cambridge University Press.
