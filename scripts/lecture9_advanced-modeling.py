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
# # Lecture 9: Advanced Modeling

# %% [markdown]
# In the previous notebook we saw which tooling to use. In this notebook we will see how to put problems in the (I)LO format. Thinking in the decision variables-objective-constraints framework often allows you to arrive at a problem formulation that resembles the standard (I)LO formulation. However, sometimes it is difficult to ensure that objective and constraints are linear. This section discusses some often-used tricks to formulate certain types of objectives and constraints in a linear way.
#
# The first trick is useful when we want to minimize the absolute value of some decision variable, thus when the problem is of the form
#
# $$
# \min\{c^T|x| \mid Ax \le b\},
# $$
#
# for some vector $c \ge 0$.
#
# :::{note} A Knapsack Problem in AMPL
# We give the AMPL implementation of the knapsack problem and a small instance which can be submitted right away to the NEOS server. The problem structure is implemented in the model file:
#
# ![AMPL model file for the knapsack problem](images/lecture9_box6.2-model.png)
#
# Next we need a data file in which the instance is given and finally the run file which tells the NEOS server what to do:
#
# ![AMPL data file for the knapsack instance](images/lecture9_box6.2-data.png)
#
# ![AMPL run file submitted to the NEOS server](images/lecture9_box6.2-run.png)
#
# Note that the problem structure and data are separated. When a planner has to solve a knapsack problem every day, he only needs to change the data file.
#
# Further details on the AMPL syntax can be found online or in Fourer, Gay, and Kernighan (2003).
# :::
#
# The crucial idea is that the variable $x_i$ can be rewritten as follows: $x_i = x_i^+ - x_i^-$ with $x_i^+, x_i^- \ge 0$ and one of them 0. Now the optimization problem can be rewritten as follows:
#
# $$
# \min\{c^T(x^+ + x^-) \mid A(x^+ - x^-) \le b,\ x^+, x^- \ge 0\},
# $$
#
# which is linear. Because $c \ge 0$ for each $i$ either $x_i^+ = 0$ or $x_i^- = 0$.
#
# :::{exercise}
# :label: ex-6-16
#
# Numbers $a_1, \dots, a_n$ are given. We are looking for $x$ that minimizes $\sum_i |x - a_i|$. Formulate this as LO problem, and implement it in Excel for the following numbers: 1, 2, 3, 5, 8, 10, 20, 35, 100. How can you interpret the outcome?
# :::
#
# The previous exercise shows that the median minimizes the sum of absolute errors, much as the average minimizes the sum of squared errors. We can extend this to linear functions. We already did this for squared errors, for which linear regression is the method. For absolute errors it is called quantile regression. Points $(x_i, y_i)$ are given and the objective is to find a function $y = a + bx$ such that the sum of absolute errors is minimized, see the figure below.

# %% [markdown]
# ![Quantile regression](images/lecture9_fig6.11.png)

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
# :::{exercise}
# :label: ex-6-17
#
# Consider [the call center staffing exercise](lecture9_ilo-applications.ipynb#ex-6-13). Assume we only have 8-hour shifts. To avoid overstaffing we replace the condition that staffing is met in every interval by the following objective: minimize the sum of absolute differences between demand and schedule. Formulate this as a LO problem and solve it using Excel.
# :::
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
# :::{exercise}
# :label: ex-6-18
#
# Consider [the transportation exercise](lecture8_linear-optimization.ipynb#ex-6-8), but with an additional feature: every link that is used has fixed costs 10. Determine the optimal solution, using ILO.
# :::
#
# :::{exercise}
# :label: ex-6-19
#
# Consider the [multi-period production/inventory model](lecture8_linear-optimization.ipynb#production-inventory-model). Extend it to fixed order costs, meaning that costs $K$ are incurred at $t$ when $x_t > 0$, keeping all constraints linear.
# :::
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
# :::{exercise}
# :label: ex-6-20
#
# Assume that activities B and C of [the project planning problem](lecture8_linear-optimization.ipynb#project-planning) use the same resource and therefore cannot be scheduled at the same time. Formulate this as ILO problem and solve it using Excel.
# :::

# %% [markdown]
# (machine-scheduling)=
# ## Machine Scheduling
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
# :::{exercise}
# :label: ex-6-21
#
# Implement the single-machine scheduling problem with tardiness as objective in AMPL. Solve it for the following data with an appropriate solver on the NEOS server:
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
# :::
#
# :::{exercise}
# :label: ex-6-22
#
# Change the objective of [the class-scheduling exercise](lecture9_ilo-applications.ipynb#ex-6-14) as follows: the time the last class finishes has to be minimized.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.7 "Modeling Tricks."
# - Fourer, R., Gay, D.M., & Kernighan, B.W. (2003). *AMPL: A Modeling Language for Mathematical Programming*. Duxbury, Thomson.
