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
# # Lecture 8: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in [Introduction](lecture8_introduction.ipynb),
# [Linear Optimization](lecture8_linear-optimization.ipynb), and
# [Integer Optimization](lecture8_integer-optimization.ipynb) check what you just read.
# This notebook collects the larger exercises for Lecture 8: independent problems worth
# more time, starting with a set of homework exercises, followed by further exercises.
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
# :label: hw-8-1
#
# A company has resources to produce 5 units of product 1 and 3.5 units of product 2.
# However, both products need labor, which is limited to 10 days. A unit of product 1
# requires 1 day of work, a unit of product 2 requires 2 days of work. The objective is to
# maximize the total number of products produced.
#
# a. Model this as a linear optimization problem (the solution does not need to be
#    integer). "Model" means writing out the model on paper with well-defined decision
#    variables used in the objective function and constraints.
#
# b. Make a graphical representation of the problem and determine the optimal solution
#    algebraically.
#
# c. Suppose the number of units of product 2 produced must be integer. Give *all* optimal
#    solutions and motivate why they are optimal.
#
# d. In exchange for 0.25 units of product 2, you can obtain 5 additional labor days; this
#    exchange is possible only once, and the number of units of product 2 produced no
#    longer has to be integer. Model this as an integer linear optimization problem, and
#    motivate whether you should make use of this exchange.
# :::
#

# %% [markdown]
#
# :::{solution} hw-8-1
# :label: sol-hw-8-1
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of product 1 and 2, respectively.
#    Then the LO is
#
#    $$
#    \max x + y \quad \text{s.t.} \quad x \le 5,\ y \le 3.5,\ x + 2y \le 10,\ x, y \ge 0.
#    $$
#
# b. The optimal solution is where lines $x = 5$ and $x + 2y = 10$ cross: $x = 5$,
#    $y = 2.5$, objective value 7.5.
#
# :::{figure} images/lecture8_hw1-ex1b.png
# :label: fig-hw-8-1b
#
# The feasible region of part a, with the optimum at $(5, 2.5)$.
# :::
#
# c. Shifting the objective line $x + y = 7.5$ toward the feasible region, the objective
#    line $x + y = 7$ first touches two feasible integer points at once: where it crosses
#    $x + 2y = 10$, giving $(x, y) = (4, 3)$, and where it crosses $x = 5$, giving
#    $(x, y) = (5, 2)$. Both have objective value 7, and since these are the first
#    feasible integer points the line reaches, both are optimal.
#
# d. Let binary $z = 1$ if the exchange is used, 0 otherwise. The ILO becomes
#
#    $$
#    \begin{aligned}
#    \max \quad & x + y - 0.25z \\
#    \text{s.t.} \quad & x \le 5,\ y \le 3.5,\ x + 2y \le 10 + 5z,\ y \ge 0.25z,\ x, y \ge 0,\ z \in \{0, 1\}.
#    \end{aligned}
#    $$
#
#    Case $z = 0$ is solved in part b, objective 7.5. For $z = 1$, the labor constraint
#    becomes $x + 2y \le 15$, and the corner of the feasible region moves to where
#    $x = 5$ meets $y = 3.5$, giving objective $5 + 3.5 - 0.25 = 8.25$. Since
#    $8.25 > 7.5$, it is optimal to use the exchange.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-8-2
#
# A bakery produces two types of bread, A and B, using only flour and yeast. Each unit of
# A needs 1 unit of flour and 2 units of yeast; each unit of B needs 1 unit of flour and 1
# unit of yeast. The bakery has 5 units of flour and 7.5 units of yeast, and makes a
# profit of 1.5 euro per unit of A and 1 euro per unit of B sold. It wants to maximize
# profit.
#
# a. Model the bakery problem as an LO problem, assuming the solution does not have to be
#    integer.
#
# b. Make a graphical representation of the problem and determine the optimal baking
#    solution algebraically.
#
# c. Suppose the profits per unit of A and B are, more generally, $p_A \ge 0$ and
#    $p_B \ge 0$. Consider two candidate baking plans: (i) 0 units of A and 5 units of B;
#    (ii) 1 unit of A and 4 units of B. For each plan, give and motivate a value for the
#    pair $(p_A, p_B)$ such that that plan is optimal.
#
# d. Now the profit for the first 2 units of A is 1.5 euro per unit, and from the 2nd unit
#    onward it drops to 0.5 euro per unit (e.g. 3.5 units of A give a profit of
#    $2 \times 1.5 + 1.5 \times 0.5 = 3.75$ euro). Model this as an LO problem and derive
#    the optimal solution.
# :::
#

# %% [markdown]
#
# :::{solution} hw-8-2
# :label: sol-hw-8-2
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of bread types A and B. Then the LO
#    is
#
#    $$
#    \max 1.5x + y \quad \text{s.t.} \quad x + y \le 5,\ 2x + y \le 7.5,\ x, y \ge 0.
#    $$
#
# b. The optimal solution is where lines $x + y = 5$ and $2x + y = 7.5$ cross: $x = 2.5$,
#    $y = 2.5$, objective value 6.25.
#
# :::{figure} images/lecture8_hw1-ex2b.png
# :label: fig-hw-8-2b
#
# The feasible region of part a, with the optimum at $(2.5, 2.5)$.
# :::
#
# c. The objective function is $p_A x + p_B y$.
#
#    i. $(p_A, p_B) = (0, 1)$: since A earns no profit, it is optimal to bake as much B as
#       possible, i.e. $(x, y) = (0, 5)$.
#
#    ii. From part b, $(x, y) = (1, 4)$ lies on the constraint line $x + y = 5$. Choosing
#        the objective line parallel to that constraint, e.g. $(p_A, p_B) = (1, 1)$, makes
#        every point on that edge, including $(1, 4)$, optimal.
#
# d. Let nonnegative $z$ be the less profitable, unrestricted continuation of $x$, and cap
#    $x$ itself at 2. The LO becomes
#
#    $$
#    \begin{aligned}
#    \max \quad & 1.5x + y + 0.5z \\
#    \text{s.t.} \quad & x + y + z \le 5,\ 2x + y + 2z \le 7.5,\ x \le 2,\ x, y, z \ge 0.
#    \end{aligned}
#    $$
#
#    :::{figure} images/lecture8_hw1-ex2d.png
#    :label: fig-hw-8-2d
#
#    The feasible region once $x$ is capped at 2 and $z$ takes over (axes read as $y$ and
#    $z$).
#    :::
#
#    The optimizer always fills the more profitable $x$ before $z$; working out both the
#    $z = 0$ and $z > 0$ cases leads to the same optimum $(x, y, z) = (2, 3, 0)$, objective
#    $1.5 \times 2 + 3 = 6$.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-8-3
#
# You are packing a knapsack with a remaining capacity of 9 kg for a road trip. Four items
# remain, with sizes (kg) $(2, 2, 4, 4)$ and rewards (euro) $(3, 1, 8, 5)$.
#
# a. Write down the integer linear optimization formulation and give the optimal solution.
#
# b. Determine the optimal solution of the LO relaxation (i.e. without the binary
#    constraints).
#
# c. Determine the optimal solution using branch and bound, and make a drawing of the
#    steps you took.
#
# d. Suppose there is now also a volume capacity of 12 cubic decimeters, and the items'
#    volumes (dm$^3$) are $(4, 8, 2, 10)$. Modify part a to account for this extra
#    constraint. What is the optimal solution?
# :::
#

# %% [markdown]
#
# :::{solution} hw-8-3
# :label: sol-hw-8-3
# :class: dropdown
#
# a. Number the items 1-4 and let $x_i = 1$ if item $i$ is taken, 0 otherwise. The ILO is
#
#    $$
#    \max 3x_1 + x_2 + 8x_3 + 5x_4 \quad \text{s.t.} \quad 2x_1 + 2x_2 + 4x_3 + 4x_4 \le 9,\
#    x_i \in \{0, 1\}.
#    $$
#
#    Checking the combinations that use the capacity fully: $\{1, 2, 3\}$ (size 8, reward
#    12), $\{1, 2, 4\}$ (size 8, reward 9), $\{3, 4\}$ (size 8, reward 13). The optimal
#    solution is $(x_1, x_2, x_3, x_4) = (0, 0, 1, 1)$, reward 13.
#
# b. The reward-to-size ratios are $1.5, 0.5, 2, 1.25$ for items 1-4, so fill the knapsack
#    in the order 3, 1, 4, 2. Items 3 and 1 fit completely (total size $4 + 2 = 6$),
#    leaving 3 kg for item 4 (size 4), i.e. a $3/4$ fraction of it. The optimal LO
#    relaxation solution is $(x_1, x_2, x_3, x_4) = (1, 0, 1, 3/4)$, objective
#    $8 + 3 + 3.75 = 14.75$.
#
# c. Branching on the fractional variables of the LO relaxation from part b, the optimum
#    is found at step 13: $(0, 0, 1, 1)$, reward 13, matching part a.
#
# :::{figure} images/lecture8_hw1-ex3c.png
# :label: fig-hw-8-3c
#
# Branch and bound applied to the knapsack problem (UB = upper bound from the LO
# relaxation, LB = lower bound from a feasible integer solution).
# :::
#
# d. Adding $4x_1 + 8x_2 + 2x_3 + 10x_4 \le 12$: the solution from part c,
#    $(0, 0, 1, 1)$, uses volume $2 + 10 = 12 \le 12$, so it remains feasible and
#    therefore still optimal: reward 13.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-8-4
#
# Two products give the same revenue per unit produced but use different quantities of 2
# resources. Per unit produced, product 1 uses 4 units of resource 1 and 2 units of
# resource 2; product 2 uses 1 and 4 units, respectively. There are 12 units of resource 1
# and 16 units of resource 2 available. We want to maximize revenue.
#
# a. Formulate the linear optimization model of this product-mix problem.
#
# b. Make a graphical representation of the problem and determine the optimal solution
#    algebraically.
#
# c. Now the quantities produced can only be integer. Which constraints do you have to
#    add? What is the optimal solution in the integer case?
#
# d. In the integer case, suppose instead that there are 20 units of resource 2. What is
#    the additional revenue compared to part c?
# :::
#

# %% [markdown]
#
# :::{solution} hw-8-4
# :label: sol-hw-8-4
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of product 1 and 2. Then
#
#    $$
#    \max x + y \quad \text{s.t.} \quad 4x + y \le 12,\ 2x + 4y \le 16,\ x, y \ge 0.
#    $$
#
# b. The optimal solution is where lines $4x + y = 12$ and $2x + 4y = 16$ cross:
#    $x = 16/7$ and $y = 20/7$, objective value $36/7 \approx 5.14$.
#
# :::{figure} images/lecture8_hw1-ex4b.png
# :label: fig-hw-8-4b
#
# The feasible region of part a, with the continuous optimum at $(16/7, 20/7)$.
# :::
#
# c. Add the constraints $x, y \in \{0, 1, 2, \dots\}$. Shifting the objective line down
#    from the continuous optimum, the first feasible integer point it reaches is
#    $(x, y) = (2, 3)$, objective value 5. So the optimal integer solution is
#    $(x, y) = (2, 3)$.
#
# d. Constraint $2x + 4y \le 16$ becomes $2x + 4y \le 20$. The continuous optimum is now
#    where $4x + y = 12$ meets $2x + 4y = 20$: $x = 2$, $y = 4$, objective value 6. Since
#    this solution is already integer, it is also optimal for the integer case. The
#    additional revenue compared to part c is $6 - 5 = 1$ euro.
# :::

# %% [markdown]
# ## Further Exercises
#
# These further exercises test your knowledge, your pulp skills, and include more
# homework-style questions: some from the book, some that need pulp (not possible on the
# exam).

# %% [markdown]
# :::{exercise}
# :label: ex-6-2
#
# Consider the following LO problem:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & 2x_1 + x_2 + 4x_3 \\
# \text{subject to} \quad & x_1 - 2x_2 + 2x_3 \le 120 \\
# & -x_1 - x_2 + 3x_3 = 100 \\
# & x_1 - x_2 + x_3 \ge 80 \\
# & x_i \ge 0 \text{ for all } i.
# \end{aligned}
# $$
#
# a. Solve it in pulp (pulp accepts `>=` and `==` constraints directly).
#
# b. Rewrite it in the general form below (maximization, only "$\le$" constraints).
#
# c. Solve the rewritten problem and check it gives the same solution as a.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-4
#
# Re-solve the larger LO problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb#larger-lo-example) adding the
# two resource constraints one at a time, and note how the optimal objective value changes
# after each addition.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-5
#
# The tax office can only check a subset of the declarations it received. There are 3 types
# of employees with different skills and 3 types of declarations. The expected extra-tax
# revenue per declaration type is (200, 1000, 500) euros. Every employee can process every
# declaration, except employee type 2 who cannot process declaration type 2 and employee
# type 3 who cannot process declaration type 3. The time per declaration is (1, 3, 2) hours,
# except employee type 3 who takes 2 hours for a type 1 declaration. The numbers of
# declarations are (15000, 6000, 8000); the numbers of available hours are
# (10000, 20000, 15000). How do you assign the employees to the declaration types? Solve
# with pulp.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-6
#
# Extend the larger LO problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb#larger-lo-example). It helps to
# ask what the additional *decision* is.
#
# a. Next to the 15 oak panels, you can buy extra ones for a price of 1 per panel. What is
#    the optimal solution now?
#
# b. The same, but the price per extra panel is 3.
#
# c. The same, but the price is 6. Can you interpret the result?
# :::
