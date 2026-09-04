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
# # Lecture 8: Integer Optimization

# %% [markdown]
# For the simplex algorithm to be used, it is essential that the objective and all constraints are linear. Many extensions exist to non-linear functions. One important class is where there is, in addition to the linear constraints, constraints requiring one or more of the decision variables to be integer (i.e., taking values in $\{0, 1, 2, \dots\}$) or binary (taking values in $\{0, 1\}$). We call these integer linear optimization (ILO) problems.
#
# Note that binary problems are special cases of integer problems: the constraint $x_i \in \{0, 1\}$ is equivalent to the following 2 constraints: $x_i \in \{0, 1, 2, \dots\}$ and $x_i \le 1$.
#
# Product-mix problems where we have to produce integer numbers of items is a good example. In R we can add an additional argument to the solver call:
#
# ```r
# > lp ("max", f.obj, f.con, f.dir, f.rhs, all.int=TRUE)
# ```
#
# In Excel we have to add additional constraints, as in the figure below.

# %% [markdown]
# ![Entering integer and binary constraints in Excel](images/lecture8_fig6.8.png)

# %% [markdown]
# :::{exercise}
# :label: ex-6-9
#
# Solve the integer version of the [product-mix problem](lecture8_linear-optimization.ipynb#problem-formulation).
# :::
#
# The archetypical binary LO problem is the knapsack problem. You have to make a selection out of a set of items. Each item has a revenue and a weight. The goal is to maximize the total revenue with a constraint on the total weight. Typical applications of the knapsack are logistics problems, for example selecting items which have to be transported in trucks, or so-called cutting problems, which arise, for example, in steel plants where you have to cuts plates in pieces of different sizes.
#
# As an example, consider a problem with total weight capacity 11. The items are as follows:
#
# | | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
# |---|---|---|---|---|---|---|---|
# | revenue | 60 | 60 | 40 | 10 | 20 | 10 | 3 |
# | weight | 3 | 5 | 4 | 1.4 | 3 | 3 | 1 |
#
# Formulated as ILO we get:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 60x_1 + 60x_2 + 40x_3 + 10x_4 + 20x_5 + 10x_6 + 3x_7 \\
# \text{subject to} \quad & 3x_1 + 5x_2 + 4x_3 + 1.4x_4 + 3x_5 + 3x_6 + x_7 \le 11 \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# Solving this using R or Excel leads to the optimum $(1, 1, 0, 0, 1, 0, 0)$ with value 140.
#
# :::{exercise}
# :label: ex-6-10
#
# Verify that this is indeed the optimal solution by solving the problem in R and Excel.
# :::
#
# Although the solver seemed to have found the optimal answer without any problems, it required much more work. This becomes apparent when we solve big real-life ILO problems with hundreds or thousands of variables. To gain more insight in how ILOs are solved, let us have a look at the figure below, where we see the steps to solve the knapsack example. We start with solving the LO relaxation, which is the problem without the integer or binary constraints (step 1). Sometimes we find an integer solution right away. Certain types of problems are even guaranteed to give integer solutions immediately. Here however $x_3$ is non-integer. Its value (150) is an upper bound to the best integer solution.

# %% [markdown]
# ![Solving an ILO problem](images/lecture8_fig6.9.png)
#
# > **Erratum applied (p. 97):** step 9 should have objective value 127.33 (not 128) and solution $(1, 0, 1, 1, 13/15, 0, 0)$ (not $(1, 0, 1, 1, 9/10, 0, 0)$).

# %% [markdown]
# Now we branch on $x_3$, and we continue with the branch $x_3 = 0$. We solve the relaxation again, but with $x_3 = 0$. We find again a non-integer solution (step 2). We continue branching until we find an integer solution in step 4 with value 133. It is called a lower bound (LB) of the optimum: perhaps there are other integer solutions with values between 133 and 150. To find out if there are any such solutions we work our way back up to make sure all branches are dealt with. In step 5, we find an integer solution that is worse than the LB. In step 6, we find a higher binary value than the LB. It becomes the new LB, and the old LB is now sub-optimal (step 7). We have dealt with the left side of the tree, we move to the right. In step 8, we find a non-integer solution, we branch on $x_2$. In step 9 and 10, we find non-integer solutions which are worse or equal than the LB. Adding constraints will not make the value higher, therefore these branches can be discarded. We have dealt with all branches, and therefore the current LB is the optimum (step 11). This algorithm is called branch-and-bound.
#
# Many LO solvers can also handle integer constraints. However, not all solvers can solve big instances. The best solvers are proprietary, notably CPLEX and Gurobi.
#
# :::{exercise}
# :label: ex-6-11
#
# Solve by branch-and-bound the knapsack problem having rewards (15, 9, 10, 5), sizes (1, 3, 5, 4) and capacity 8. Check the result with R.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.4 "Integer Problems."
