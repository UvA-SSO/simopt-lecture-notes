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
# time, starting with a set of homework exercises, followed by further exercises.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::

# %% [markdown]
# ## Homework Exercises

# %% [markdown]
# :::{exercise}
# :label: hw-9-1
#
# A manager has to allocate 3 workers to 2 tasks: how many hours each worker spends on
# each task. The total working hours available for the workers are 20, 20, and 5,
# respectively. The tasks require 30 and 40 *effective* working hours, respectively. The
# effective working hour rate indicates how many effective hours a specific worker
# contributes to a specific task per hour worked:
#
# | Worker \ Task | 1 | 2 |
# |---|---|---|
# | 1 | 0.1 | 10 |
# | 2 | 2 | 2 |
# | 3 | 5 | 8 |
#
# For example, each hour worked by worker 1 on task 2 gives 10 effective working hours.
# The manager wants to minimize the total hours worked while ensuring that (i) both tasks
# are done, and (ii) no worker's total hours are exceeded.
#
# a. Model this as a linear optimization problem.
#
# b. Give a restriction that ensures worker 1 works at most 5 hours on task 2.
#
# c. Now only one worker may work on task 1. Model this as an integer linear optimization
#    problem.
#
# d. The restriction from part c is lifted again. Instead, if workers 2 and 3 *both* work
#    on task 1, 10 extra effective working hours are required for task 1 (due to
#    coordination overhead). Model this as an integer linear optimization problem.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-1
# :label: sol-hw-9-1
# :class: dropdown
#
# a. Let $x_{ij}$ be the number of hours worker $i$ works on task $j$. The LO is
#
#    $$
#    \begin{aligned}
#    \min \quad & \sum_{i,j} x_{ij} \\
#    \text{s.t.} \quad & x_{11} + x_{12} \le 20 \\
#    & x_{21} + x_{22} \le 20 \\
#    & x_{31} + x_{32} \le 5 \\
#    & 0.1x_{11} + 2x_{21} + 5x_{31} \ge 30 \\
#    & 10x_{12} + 2x_{22} + 8x_{32} \ge 40 \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
# b. The restriction is $x_{12} \le 5$.
#
# c. Let binary $z_i = 1$ if worker $i$ works on task 1, 0 otherwise, and let $M$ be a big
#    constant (e.g. $M = 1000$). Add to part a:
#
#    $$
#    x_{i1} \le M z_i \text{ for all } i, \qquad \sum_i z_i = 1, \qquad z_i \in \{0, 1\}.
#    $$
#
# d. Starting from part a, let binary $q = 1$ if workers 2 and 3 both work on task 1.
#    First, replace the task-1 constraint with
#
#    $$
#    0.1x_{11} + 2x_{21} + 5x_{31} \ge 30 + 10q,
#    $$
#
#    then add the $z_i$/$M$ restrictions from part c for $i = 2, 3$ (so $z_2, z_3$ track
#    whether workers 2 and 3 work on task 1), and finally add
#
#    $$
#    q \ge z_2 + z_3 - 1, \qquad q \in \{0, 1\}.
#    $$
#
#    This forces $q = 1$ once $z_2 = z_3 = 1$; otherwise, minimizing the objective already
#    pushes $q$ down to 0 (less work required is always at least as good).
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-9-2
#
# A hospital has 4 candidate locations, A, B, C, D, to place an ambulance so as to cover 5
# cities, numbered 1-5. The travel time (minutes) from each location to each city is:
#
# | Location \ City | 1 | 2 | 3 | 4 | 5 |
# |---|---|---|---|---|---|
# | A | 3 | 4 | 8 | 10 | 8 |
# | B | 8 | 5 | 4 | 10 | 12 |
# | C | 6 | 9 | 8 | 7 | 3 |
# | D | 8 | 7 | 4 | 2 | 6 |
#
# At each location, at most one ambulance can be placed. The hospital wants to minimize
# the number of ambulances placed while ensuring every city is reachable within 5 minutes
# or less from at least one location with a placed ambulance.
#
# a. Describe how this problem is a set cover problem: give the universe and the sets.
#
# b. Model the problem as an integer linear optimization problem.
#
# c. Using part b, is placing ambulances at locations A and C only a feasible allocation?
#    Motivate your answer.
#
# d. Give the optimal solution and motivate why it is optimal.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-2
# :label: sol-hw-9-2
# :class: dropdown
#
# a. The universe is $U = \{1, 2, 3, 4, 5\}$, the 5 cities. For each location $i$, let
#    $S_i$ be the set of cities reachable within 5 minutes: $S_A = \{1, 2\}$,
#    $S_B = \{2, 3\}$, $S_C = \{5\}$, $S_D = \{3, 4\}$. Minimizing the number of sets that
#    together cover $U$ is exactly the ambulance placement problem.
#
# b. Let binary $x_i = 1$ if an ambulance is placed at location $i$, for $i \in \{A, B, C,
#    D\}$. Then
#
#    $$
#    \begin{aligned}
#    \min \quad & x_A + x_B + x_C + x_D \\
#    \text{s.t.} \quad & x_A \ge 1 \ \text{(city 1)}, \quad x_A + x_B \ge 1 \ \text{(city 2)}, \\
#    & x_B + x_D \ge 1 \ \text{(city 3)}, \quad x_D \ge 1 \ \text{(city 4)}, \quad x_C \ge 1 \ \text{(city 5)}, \\
#    & x_A, x_B, x_C, x_D \in \{0, 1\}.
#    \end{aligned}
#    $$
#
# c. Setting $x_A = x_C = 1$, $x_B = x_D = 0$ violates the constraints for cities 3 and 4
#    (both require $x_B + x_D \ge 1$ or $x_D \ge 1$, which fail). So this allocation is
#    infeasible.
#
# d. Covering cities 1, 4, and 5 forces $x_A = x_D = x_C = 1$ directly from their
#    single-variable constraints, so at least 3 ambulances are required. Setting
#    $x_B = 0$ (with $x_A = x_C = x_D = 1$) is feasible (city 2 is still covered via
#    $x_A$, city 3 via $x_D$) and achieves this minimum of 3, so it is optimal: place
#    ambulances at A, C, and D.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-9-3
#
# :::{figure} images/lecture9_hw1-ex3-diagram.png
# :label: fig-hw-9-3
#
# A transportation problem with 3 sources (supply 30, 10, 10) and 2 destinations (demand
# 25, 25); arc labels are the costs per unit shipped.
# :::
#
# a. Give the LO formulation of this problem.
#
# b. Extend this formulation so that at most 3 arcs are used in total.
#
# A number of first-year students need to be assigned to rooms in student dormitories:
# there are $n$ students and $n$ rooms, and each student ranks the rooms.
#
# c. Formulate this as a transportation problem: give the sources, destinations, and a
#    reasonable cost function.
#
# d. Change the formulation so it can be used to decide whether it is possible to
#    accommodate everyone according to their top-3 preferences. Explain your answer.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-3
# :label: sol-hw-9-3
# :class: dropdown
#
# a. Let $x_{ij}$ be the quantity transported from source $i$ to destination $j$, for
#    $i = 1, 2, 3$ and $j = 1, 2$. The LO is
#
#    $$
#    \begin{aligned}
#    \min \quad & x_{11} + 2x_{12} + 3x_{21} + 5x_{22} + 5x_{31} + 3x_{32} \\
#    \text{s.t.} \quad & x_{11} + x_{12} \le 30 \ \text{(supply 1)}, \quad x_{21} + x_{22} \le 10 \ \text{(supply 2)}, \quad x_{31} + x_{32} \le 10 \ \text{(supply 3)} \\
#    & x_{11} + x_{21} + x_{31} \ge 25 \ \text{(demand 1)}, \quad x_{12} + x_{22} + x_{32} \ge 25 \ \text{(demand 2)} \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
#    Since total supply equals total demand (50 = 50), every $\le$/$\ge$ pair here could
#    equivalently be written as an equality.
#
# b. Let binary $z_{ij} = 1$ when arc $(i, j)$ is used, 0 otherwise, and let $M$ be a big
#    constant. Add $x_{ij} \le M z_{ij}$ for all $i, j$, and
#    $\sum_{i=1}^{3}\sum_{j=1}^{2} z_{ij} \le 3$.
#
# c. The sources are the students, the destinations are the rooms; supply and demand are 1
#    at every source and destination. The cost from student $i$ to room $j$ is student
#    $i$'s rank of room $j$ (so a lower rank, a more preferred room, costs less).
#
# d. Redefine the cost of arc $(i, j)$ as 0 if room $j$ is in student $i$'s top 3, and 1
#    otherwise. Solve the transportation problem: if the total cost is 0, the solution
#    gives a feasible assignment respecting everyone's top-3 preferences; if the total
#    cost is positive, no such assignment exists.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-9-4
#
# Consider the set cover problem with universe $U = \{A, B, C, D\}$ and sets
# $S_1 = \{A, B, C\}$, $S_2 = \{A, B, D\}$, $S_3 = \{A, C, D\}$, $S_4 = \{B, C, D\}$.
#
# a. Give an optimal solution and a simple but convincing argument for why it is optimal.
#
# b. Formulate the problem as an ILO.
#
# c. Formulate the LO relaxation and give an optimal solution.
#
# Now consider the *covering* problem with the same sets, costs $c = (1, 1, 1, 1)$, and
# per-element requirements $b = (3, 4, 5, 6)$ (for $A, B, C, D$ respectively).
#
# d. Give an optimal solution and a convincing argument for why it is optimal.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-4
# :label: sol-hw-9-4
# :class: dropdown
#
# a. Any choice of 2 sets is optimal: each set of size 3 is missing exactly one element
#    (a different one for each set), so no single set covers $U$, meaning 2 is the
#    minimum; and any two sets between them are missing at most the elements each is
#    individually missing, which are always covered by the other set.
#
# b. Let binary $x_i = 1$ if set $i$ is chosen, for $i = 1, 2, 3, 4$. Then
#
#    $$
#    \begin{aligned}
#    \min \quad & x_1 + x_2 + x_3 + x_4 \\
#    \text{s.t.} \quad & x_1 + x_2 + x_3 \ge 1 \ \text{(cover A)}, \quad x_1 + x_2 + x_4 \ge 1 \ \text{(cover B)}, \\
#    & x_1 + x_3 + x_4 \ge 1 \ \text{(cover C)}, \quad x_2 + x_3 + x_4 \ge 1 \ \text{(cover D)}, \\
#    & x_i \in \{0, 1\} \text{ for } i = 1, 2, 3, 4.
#    \end{aligned}
#    $$
#
# c. The LO relaxation replaces $x_i \in \{0, 1\}$ with $0 \le x_i \le 1$. By symmetry, the
#    optimal solution is $x_i = \tfrac{1}{3}$ for all $i$: every covering constraint sums
#    exactly 3 of the 4 variables, so each is satisfied with equality, at objective value
#    $4/3$.
#
# d. The covering constraints are $x_1 + x_2 + x_3 \ge 3$, $x_1 + x_2 + x_4 \ge 4$,
#    $x_1 + x_3 + x_4 \ge 5$, $x_2 + x_3 + x_4 \ge 6$. Adding all four and dividing by 3
#    gives $x_1 + x_2 + x_3 + x_4 \ge 6$, so the objective can never beat 6. The solution
#    $x = (0, 1, 2, 3)$ has objective value 6 and satisfies every constraint with equality
#    (check: $0+1+2=3$, $0+1+3=4$, $0+2+3=5$, $1+2+3=6$), so it is optimal.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-9-5
#
# For a classroom assignment, couples need to be formed among $n$ students. Each student
# lists the students they are willing to work with; a couple can only be formed if both
# students listed each other. Each student can be part of at most one couple, but not
# every student needs to be paired. Formulate an integer linear optimization model that
# maximizes the number of couples formed.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-5
# :label: sol-hw-9-5
# :class: dropdown
#
# Let parameter $e_{ij} = 1$ if student $i$ wants to work with $j$ ($i \ne j$), 0
# otherwise, and let binary $x_{ij} = 1$ if couple $(i, j)$ is formed. A first, symmetric
# formulation is
#
# $$
# \begin{aligned}
# \max \quad & \frac{1}{2}\sum_{i=1}^{n} \sum_{\substack{j=1 \\ j \ne i}}^{n} x_{ij} \\
# \text{s.t.} \quad & x_{ij} = x_{ji} \text{ for all } i \ne j \\
# & x_{ij} \le e_{ij} \text{ for all } i \ne j \\
# & \sum_{\substack{j=1 \\ j \ne i}}^{n} x_{ij} \le 1 \text{ for all } i \\
# & x_{ij} \in \{0, 1\} \text{ for all } i \ne j.
# \end{aligned}
# $$
#
# The objective is divided by 2 since each couple $(i, j)$ is counted twice (as $x_{ij}$
# and $x_{ji}$). The first constraint enforces symmetry, the second only allows feasible
# couples, and the third caps every student at one couple.
#
# Since $x_{ij} = x_{ji}$, it suffices to optimize only over $i < j$, using
# $e_{ij} = 1$ when $i$ and $j$ are willing to work together. This gives a tighter
# formulation (fewer variables and constraints):
#
# $$
# \begin{aligned}
# \max \quad & \sum_{i=1}^{n} \sum_{j=i+1}^{n} x_{ij} \\
# \text{s.t.} \quad & x_{ij} \le e_{ij} \text{ for } i < j \\
# & \sum_{j=1}^{i-1} x_{ji} + \sum_{k=i+1}^{n} x_{ik} \le 1 \text{ for all } i \\
# & x_{ij} \in \{0, 1\} \text{ for } i < j.
# \end{aligned}
# $$
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-9-6
#
# For one school day, classes must be assigned to professors so that each class gets one
# hour with each professor it needs to see, with no conflicts. A matrix $a(c, p) = 1$
# indicates that class $c$ needs to see professor $p$ (0 otherwise). Every class is at
# school from the beginning of the first hour, and stays until right after the last hour
# in which it sees a professor.
#
# a. Formulate an integer linear optimization model that minimizes the total number of
#    hours classes spend at school.
#
# b. Change the objective of part a to: minimize the time at which the *last* class is
#    ready to leave.
# :::
#

# %% [markdown]
#
# :::{solution} hw-9-6
# :label: sol-hw-9-6
# :class: dropdown
#
# a. Let binary $x_{tcp} = 1$ if class $c$ sees professor $p$ at time $t$, 0 otherwise, and
#    binary $y_{tc} = 1$ if class $c$ is at school at time $t$, 0 otherwise. The objective
#    is $\min \sum_{t,c} y_{tc}$, subject to:
#
#    - class $c$ sees professor $p$ whenever needed: $\sum_t x_{tcp} \ge a(c, p)$ for all
#      $c, p$;
#    - a class sees at most one professor at a time: $\sum_p x_{tcp} \le 1$ for all $t, c$;
#    - a professor sees at most one class at a time: $\sum_c x_{tcp} \le 1$ for all $t, p$;
#    - $y_{tc}$ gets its intended meaning: if a class sees a professor it must be at
#      school, $y_{tc} \ge \sum_p x_{tcp}$ for all $t, c$; and a class stays until right
#      after its last professor hour, $y_{tc} \ge y_{t+1,c}$ for all $t, c$;
#    - $x_{tcp}, y_{tc} \in \{0, 1\}$ for all $t, c, p$.
#
# b. Let $z$ be the latest time any class is ready to leave; the new objective is
#    $\min z$. Since $\sum_t y_{tc}$ equals the time class $c$ is ready, adding
#    $z \ge \sum_t y_{tc}$ for all $c$ to the model of part a makes minimizing $z$
#    correctly capture this.
# :::

# %% [markdown]
# ## Further Exercises
#
# These further exercises test your knowledge, your pulp skills, and include more
# homework-style questions: some from the book, some that need pulp (not possible on the
# exam).

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
