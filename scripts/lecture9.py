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
# # Lecture 9: Applications of Integer Linear Optimization; Advanced Modeling

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 6 — Linear Optimization (§6.5, §6.7)

# %% [markdown]
# #### 6.5 Example ILO problems
#
# In this section we discuss a number of problems that can be solved with ILO. The first is the set cover problem. We have a so-called universe $U = \{1, \dots, m\}$, and sets $S_1, \dots, S_n$, with $S_i \subset U$. We are looking for the smallest selection of sets that covers $U$.
#
# As an example, let $U$ be set of locations where incidents can happen, and every set $S_i$ the set of locations that can be reached by an ambulance from a certain base station within a certain target time. Then the set cover problem is as follows: what is the minimum number of ambulances needed and what are their base stations such that all locations can be reached within the target time?
#
# The ILO formulation is as follows:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} x_i \\
# \text{subject to} \quad & \sum_{i: u \in S_i} x_i \ge 1 \text{ for all } u \in U; \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# The binary constraints are necessary, as the following example shows. Let $U = \{1, 2, 3\}$ and $S_1 = \{1, 2\}$, $S_2 = \{1, 3\}$, and $S_3 = \{2, 3\}$. Then any combination of 2 sets is optimal but the LO relaxation has optimal value 1.5 with solution $(0.5, 0.5, 0.5)$.
#
# The main constraint is often replaced by the following more convenient notation: $\sum_{j=1}^{n} a_{uj} x_j \ge 1$ with $a_{uj} = 1$ if $u \in S_j$, 0 otherwise.
#
# > **Erratum applied (p. 98):** the summation index is $x_j$, not $x_i$ (the book prints $\sum_{j=1}^{n} a_{uj} x_i$).
#
# **Exercise 6.12** Solve the following ILO problem, inspired by [13]. A swimming pool is open during 12 hours, and the lifeguards at duty should be selected. Every lifeguard has his/her own working hours and wage, as given in the table. Select the optimal combination of lifeguards assuring at least 1 lifeguard at every moment. The hours mentioned are the first and last hour that each lifeguards works, thus Ben/Celia/Fred is a feasible solution.
#
# | Lifeguard | Ann | Ben | Celia | Dick | Estelle | Fred |
# |---|---|---|---|---|---|---|
# | Hours | 1-6 | 1-4 | 5-8 | 7-10 | 7-12 | 9-12 |
# | Wages | 8 | 6 | 6 | 3 | 7 | 3 |
#
# Solve it in Excel and use the SUMPRODUCT function and a matrix with the values of $a_{ui}$.
#
# A generalization of the set cover problem is the covering problem. Instead of having to cover each element of the universe by 1 it can be more general. This leads to the following problem formulation:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} x_i \\
# \text{subject to} \quad & \sum_{i=1}^{n} a_{ui} x_i \ge b_u \text{ for all } u \in U; \\
# & x_i \in \mathbb{N}_0 = \{0, 1, 2, \dots\} \text{ for all } i.
# \end{aligned}
# $$
#
# This problem can be applied to shift scheduling, a problem already introduced by Dantzig in [7]. In shift scheduling, we have to find the best combination of shifts of employees, which in total cover the required workforce in every time interval. The seminal problem studied by Dantzig involved employees at a toll station, where the required coverage fluctuates during the day.
#
# To model this as a covering problem, let $U$ be the set of time intervals. Every set $S_i$ corresponds to a shift (with $a_{ui} = 1$ meaning shift $i$ works at time $u$), and $b_u$ the number of required workers at time $u$. Then $x_i$ corresponds to the number of employees that need to have shift $i$. By adding a coefficient to $x_i$ in the objective we can add different costs to the shifts.
#
# **Exercise 6.13** The required staffing in a call center, from 9am to 9pm in 30-minute intervals, is as follows:
#
# 10, 11, 13, 16, 16, 13, 11, 10, 10, 11, 12, 13, 14, 14, 13, 11, 10, 9, 9, 10, 9, 8, 8, 8.
#
# There are 2 types of shifts:
#
# - 8 hours working time, with a 30-minute unpaid break in the middle, wage 20 Euro/hr, possible starting times every half hour from 9am to 12:30pm;
# - 4 hours consecutive, wage 24 Euro/hr, possible starting times every half hour from 9am to 5pm.
#
# Formulate this as a covering problem and solve it with the Excel solver.
#
# Machine scheduling is another important class of ILO problems. However, because it involves some modeling tricks that are discussed in Section 6.7, we defer discussing it to that section.
#
# The next two exercises concern problems that can be solved with appropriately chosen binary decision variables.
#
# **Exercise 6.14** For a day at a school, classes have to be assigned to professors such that each class has an hour with each required professor and such that there are no conflicts such as a professor having to teach two classes at the same time. A matrix with entries $a_{cp}$ indicates which classes need to have which professors: when $a_{cp} = 1$ then class $c$ needs to have professor $p$, otherwise $a_{cp} = 0$.
#
# Formulate an ILO model that minimizes the total number of hours that classes have to spend at school. A class remains at school until right after the last hour it has seen a professor.
#
# **Exercise 6.15** For a classroom assignment, pairs need to be made of $n$ students. Each student can give a list of students he or she is willing to work with. Formulate an ILO model that maximizes the number of pairs that can be made. Each student is only allowed to be part of one pair, but it might not be possible to assign all students to a pair.

# %% [markdown]
# #### 6.7 Modeling tricks
#
# In the previous section we saw which tooling to use. In this section we will see how to put problems in the (I)LO format. Thinking in the decision variables-objective-constraints framework often allows you to arrive at a problem formulation that resembles the standard (I)LO formulation. However, sometimes it is difficult to ensure that objective and constraints are linear. This section discusses some often-used tricks to formulate certain types of objectives and constraints in a linear way.
#
# The first trick is useful when we want to minimize the absolute value of some decision variable, thus when the problem is of the form
#
# $$
# \min\{c^T|x| \mid Ax \le b\},
# $$
#
# for some vector $c \ge 0$.
#
# > **Box 6.2. A knapsack problem in AMPL**
# >
# > We give the AMPL implementation of the knapsack problem and a small instance which can be submitted right away to the NEOS server. The problem structure is implemented in the model file:
#
# ![Box 6.2 (model file): AMPL model file for the knapsack problem](images/lecture9_box6.2-model.png)
#
# > Next we need a data file in which the instance is given and finally the run file which tells the NEOS server what to do:
#
# ![Box 6.2 (data file): AMPL data file for the knapsack instance](images/lecture9_box6.2-data.png)
#
# ![Box 6.2 (run file): AMPL run file submitted to the NEOS server](images/lecture9_box6.2-run.png)
#
# > Note that the problem structure and data are separated. When a planner has to solve a knapsack problem every day, he only needs to change the data file.
# >
# > Further details on the AMPL syntax can be found online or in the AMPL book [10].
#
# The crucial idea is that the variable $x_i$ can be rewritten as follows: $x_i = x_i^+ - x_i^-$ with $x_i^+, x_i^- \ge 0$ and one of them 0. Now the optimization problem can be rewritten as follows:
#
# $$
# \min\{c^T(x^+ + x^-) \mid A(x^+ - x^-) \le b,\ x^+, x^- \ge 0\},
# $$
#
# which is linear. Because $c \ge 0$ for each $i$ either $x_i^+ = 0$ or $x_i^- = 0$.
#
# **Exercise 6.16** Numbers $a_1, \dots, a_n$ are given. We are looking for $x$ that minimizes $\sum_i |x - a_i|$. Formulate this as LO problem, and implement it in Excel for the following numbers: 1, 2, 3, 5, 8, 10, 20, 35, 100. How can you interpret the outcome?
#
# The previous exercise shows that the median minimizes the sum of absolute errors, much as the average minimizes the sum of squared errors. We can extend this to linear functions. We already did this for squared errors, for which linear regression is the method. For absolute errors it is called quantile regression. Points $(x_i, y_i)$ are given and the objective is to find a function $y = a + bx$ such that the sum of absolute errors is minimized, see Figure 6.11.

# %% [markdown]
# ![Figure 6.11: Quantile regression](images/lecture9_fig6.11.png)

# %% [markdown]
# In vector notation the problem can be formulated as follows:
#
# $$
# \min\{\mathbf{1}^T|e| \mid y - (a\mathbf{1} + bx) = e\},
# $$
#
# with $\mathbf{1}$ a vector with only 1's, and $e_i = y_i - (a + bx_i)$ the errors as given in the constraint. This can be made linear as follows:
#
# $$
# \min\{\mathbf{1}^T(e^+ + e^-) \mid y - (a\mathbf{1} + bx) = e^+ - e^-,\ e^+, e^- \ge 0\}.
# $$
#
# We can generalize this to an asymmetric objective in the following way:
#
# $$
# \min\{p\mathbf{1}^Te^+ + (1-p)\mathbf{1}^Te^- \mid y - (a\mathbf{1} + bx) = e^+ - e^-,\ e^+, e^- \ge 0\},
# $$
#
# with $0 < p < 1$. This explains the term quantile regression.
#
# **Exercise 6.17** Consider Exercise 6.13. Assume we only have 8-hour shifts. To avoid overstaffing we replace the condition that staffing is met in every interval by the following objective: minimize the sum of absolute differences between demand and schedule. Formulate this as a LO problem and solve it using Excel.
#
# The next modeling trick is for cases where the objective function is not a sum but a maximum. The full problem is then of the form
#
# $$
# \min\{\max\{x_1, \dots, x_n\} \mid Ax \le b,\ x \ge 0\}.
# $$
#
# When discussing project planning we already say how this can be put in the LO framework:
#
# $$
# \min\{z \mid z\mathbf{1} \ge x,\ Ax \le b,\ x \ge 0\}.
# $$
#
# Finally, there are some uses of a very big number, often called big M, in the context of ILO. The first is when an indicator function is part of the objective. An indicator function is a 0/1 function which is equal to 1 when a condition is satisfied. As an example, take the transportation problem with fixed costs $K$ when a link is used. To model this, we introduce binary variables $y_{ij}$ such that $y_{ij} = 1 \Leftrightarrow x_{ij} > 0$. Now we can simply add $K \sum_{i,j} y_{ij}$ to the objective function. But how to set $y_{ij}$? Here the value $M$ comes into play, by adding the following constraints:
#
# $$
# M y_{ij} \ge x_{ij}.
# $$
#
# We assume $M$ is bigger than any $x_{ij}$ ever can be. Thus, $x_{ij} > 0 \Rightarrow M y_{ij} > 0 \Rightarrow y_{ij} = 1$. When $x_{ij} = 0$ then $y_{ij}$ can be 0 or 1. Because we are minimizing costs and $K > 0$, $y_{ij}$ will be 0. Thus $y_{ij} = 1 \Leftrightarrow x_{ij} > 0$.
#
# **Exercise 6.18** Consider Exercise 6.8, but with an additional feature: every link that is used has fixed costs 10. Determine the optimal solution, using ILO.
#
# **Exercise 6.19** Consider the multi-period production/inventory model of page 94. Extend it to fixed order costs, meaning that costs $K$ are incurred at $t$ when $x_t > 0$, keeping all constraints linear.
#
# Big M can also be used in other situations, notably when a constraint only has to hold when a condition, which is part of the decision variables is satisfied. This condition is represented by a binary variable, let's say $y$, and the constraint is of the form $x \le b$. Then a linear implementation is $x \le b + (1-y)M$. When $y = 0$ the contraint always holds because the right-hand side is very big. When $y = 1$ then we find the original $x \le b$.
#
# As an example, consider 2 jobs, A and B, that have to be executed consecutively, but the order is a decision to be made. Let $x_A$ ($x_B$) be the starting time of job A (B), and $d_A$ ($d_B$) the duration of A (B). When A goes before B then we get the condition $x_A + d_A \le x_B$, otherwise $x_B + d_B \le x_A$. This can be modeled in a linear way by having the following conditions:
#
# $$
# \begin{aligned}
# x_A + d_A &\le x_B + (1-y)M, \\
# x_B + d_B &\le x_A + yM, \\
# y &\in \{0, 1\}.
# \end{aligned}
# $$
#
# Here $y = 1$ corresponds to A before B.
#
# **Exercise 6.20** Assume that activities B and C of the project planning problem of Figure 6.5 use the same resource and therefore cannot be scheduled at the same time. Formulate this as ILO problem and solve it using Excel.
#
# **Machine scheduling**
#
# A problem in which several of these concepts occur is machine scheduling. We consider jobs that need to be scheduled on a single machine. Job $i$ has release date $r_i$ before which it cannot start, duration $d_i$, and due date $t_i$, $i = 1, \dots, n$. The decision variable are $x_i$, when to start job $i$, and also binary variables $y_{ij}$ with $y_{ij} = 1$ iff (read: if and only if) job $i$ goes before $j$, for all $i \ne j$. We give all constraints and discuss objectives right after that:
#
# $$
# \begin{aligned}
# x_i + d_i &\le x_j + M y_{ji} \text{ for all } i, j \text{ such that } i \ne j,\ M \gg 0 & \text{(overlap)} \\
# y_{ij} + y_{ji} &= 1 \text{ for all } i, j \text{ such that } i \ne j & \text{(order)} \\
# x_i &\ge r_i \text{ for all } i & \text{(release dates)} \\
# y_{ij} &\in \{0, 1\} \text{ for all } i, j \text{ such that } i \ne j
# \end{aligned}
# $$
#
# Different objectives are possible. Some common ones are:
#
# - the flowtime, defined as $\sum_i (x_i + d_i - r_i)$, thus the sum of the times that the job is waiting to be processed or is being processed; that is, the time they spend "in the system";
# - the makespan, defined as $\max_i \{x_i + d_i\}$, the time when the machine is ready with all jobs;
# - the total tardiness, which is $\sum_i (x_i + d_i - t_i)^+$, the sum of the times that the jobs are late, counting finishing early as 0.
#
# The makespan can be modeled by an additional decision variable $z$ that needs to be minimized and which needs to satisfy: $x_i + d_i \le z$ for all $i$.
#
# The total tardiness can be modeled using additional decision variables $z_i$ representing the tardiness of job $i$. The objective becomes $\min \sum_i z_i$, and we get additional constraints $x_i + d_i - t_i \le z_i$ and $z_i \ge 0$ for all $i$.
#
# The model quickly becomes big, also for moderate $n$: the number of variables is $n^2$, and the number of constraints is $2n^2 + n$.
#
# **Exercise 6.21** Implement the single-machine scheduling problem with tardiness as objective in AMPL. Solve it for the following data with an appropriate solver on the NEOS server:
#
# | | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
# |---|---|---|---|---|---|---|---|---|---|---|
# | duration | 4 | 5 | 3 | 5 | 7 | 1 | 0 | 3 | 2 | 10 |
# | release time | 3 | 4 | 7 | 11 | 10 | 0 | 0 | 10 | 0 | 15 |
# | due date | 11 | 12 | 20 | 25 | 20 | 10 | 30 | 30 | 10 | 20 |
#
# To implement a constraint that needs to hold for all $i, j$ with $i \ne j$ you can use the following AMPL syntax:
#
# ```
# subject to example_constraint {i in 1..N, j in 1..M: i<>j}:
# ```
#
# 2-dimensional binary variables are defined as follows:
#
# ```
# var x {1..N, 1..M} binary;
# ```
#
# **Exercise 6.22** Change the objective of Exercise 6.14 as follows: the time the last class finishes has to be minimized.

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Applications of (integer) linear optimization (§6.5); Advanced modeling (§6.7)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — §6.5 "Example ILO problems"; §6.7 "Modeling tricks"
# - **Errata applied** (per `Literature/Erratum Book An Introduction to Business Analytics by Koole (2019).pdf`)**:**
#   - p. 98 (§6.5, set cover): the convenient-notation constraint reads $\sum_{j=1}^{n} a_{uj} x_j \ge 1$ — the book prints $x_i$ instead of $x_j$.
