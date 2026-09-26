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

# %% tags=["remove-cell"]
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def draw_line(ax, a, b, c, xlim, **style):
    """Draw the line a x + b y = c over the horizontal range xlim."""
    if b == 0:
        ax.axvline(c / a, **style)
    else:
        xs = [xlim[0], xlim[1]]
        ax.plot(xs, [(c - a * x) / b for x in xs], **style)


def draw_region(ax, corners, xlim, ylim, xlabel="$x$", ylabel="$y$"):
    """Shade the feasible region with the given corner points."""
    ax.add_patch(Polygon(corners, alpha=0.15, label="feasible region"))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# %% tags=["remove-cell"] label="hw-8-1b"
fig, ax = plt.subplots(figsize=(5, 4))
lim_x, lim_y = (0.0, 8.0), (0.0, 5.5)
draw_region(ax, [(0, 0), (5, 0), (5, 2.5), (3, 3.5), (0, 3.5)], lim_x, lim_y)
draw_line(ax, 1, 0, 5, lim_x, color="C0", label=r"$x \leq 5$")
draw_line(ax, 0, 1, 3.5, lim_x, color="C1", label=r"$y \leq 3.5$")
draw_line(ax, 1, 2, 10, lim_x, color="C2", label=r"$x + 2y \leq 10$")
draw_line(ax, 1, 1, 7.5, lim_x, color="k", ls="--", label="$x + y = 7.5$")
ax.plot(5, 2.5, "C3*", markersize=14, zorder=4, clip_on=False, label="optimum")
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% tags=["remove-cell"] label="hw-8-1c"
fig, ax = plt.subplots(figsize=(5, 4))
draw_region(ax, [(0, 0), (5, 0), (5, 2.5), (3, 3.5), (0, 3.5)], lim_x, lim_y)
for level in range(4):
    x_max = min(5, 10 - 2 * level)
    segment_label = "feasible ($y$ integer)" if level == 0 else None
    ax.plot(
        [0, x_max],
        [level, level],
        color="C2",
        lw=3,
        label=segment_label,
        clip_on=False,
    )
draw_line(ax, 1, 1, 7, lim_x, color="k", ls="--", label="$x + y = 7$")
ax.plot(
    5,
    2.5,
    "C3*",
    markersize=14,
    zorder=4,
    clip_on=False,
    label="LO relaxation optimum",
)
ax.plot(
    [5, 4],
    [2, 3],
    "ko",
    markersize=8,
    zorder=4,
    clip_on=False,
    label="optimal solutions",
)
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% [markdown]
#
# ::::{solution} hw-8-1
# :label: sol-hw-8-1
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of product 1 and 2, respectively.
#    Then the LO is
#
#    $$
#    \begin{aligned}
#    \max \quad & x + y \\
#    \text{s.t.} \quad & x \le 5 \\
#    & y \le 3.5 \\
#    & x + 2y \le 10 \\
#    & x, y \ge 0.
#    \end{aligned}
#    $$
#
# b. The figure below shows the feasible region. Shifting the objective line $x + y = c$
#    up (increasing $c$) as far as possible, it last touches the feasible region where the
#    lines $x = 5$ and $x + 2y = 10$ cross. Substituting $x = 5$ in $x + 2y = 10$ gives
#    $y = 2.5$: the optimal solution is $(x, y) = (5, 2.5)$ with objective value 7.5.
#
# :::{figure} #hw-8-1b
# :label: fig-hw-8-1b
#
# The feasible region of part a, with the objective line $x + y = 7.5$ through the optimum
# $(5, 2.5)$.
# :::
#
# c. Only $y$ has to be integer; $x$ may still take any value. We use
#    [branch and bound](lecture8_integer-optimization.ipynb#branch-and-bound). The LO
#    relaxation is the problem of part b, with optimum $(5, 2.5)$ and UB 7.5. Since
#    $y = 2.5$ is not integer, we branch on $y$:
#
#    - $y \le 2$: the LO optimum is $(5, 2)$ with value 7. Here $y$ is integer, so this is
#      a feasible solution and gives LB 7. It is also the only optimal solution of this
#      subproblem, since $x \le 5$ and $y \le 2$.
#    - $y \ge 3$: now $3 \le y \le 3.5$, and the labor constraint gives $x \le 10 - 2y$,
#      so the objective is at most $10 - y$. The LO optimum is therefore $(4, 3)$ with
#      value 7, again with integer $y$, and no other point of this subproblem reaches 7.
#
#    Both subproblems are solved and the pool is empty, so the optimal value is 7, and it
#    is reached in both subproblems. The optimal solutions are $(x, y) = (5, 2)$ and
#    $(x, y) = (4, 3)$. The figure below shows the same result graphically: with integer
#    $y$, the feasible solutions are the horizontal line segments, and the objective line
#    $x + y = 7$ touches them in exactly these two points.
#
# :::{figure} #hw-8-1c
# :label: fig-hw-8-1c
#
# With $y$ integer and $x$ continuous, the feasible solutions of part c lie on the
# horizontal segments. The objective line $x + y = 7$ touches them in $(5, 2)$ and
# $(4, 3)$.
# :::
#
# d. Let binary $z = 1$ if the exchange is used, 0 otherwise. The ILO becomes
#
#    $$
#    \begin{aligned}
#    \max \quad & x + y - 0.25z \\
#    \text{s.t.} \quad & x \le 5 \\
#    & y \le 3.5 \\
#    & x + 2y \le 10 + 5z \\
#    & y \ge 0.25z \\
#    & x, y \ge 0 \\
#    & z \in \{0, 1\}.
#    \end{aligned}
#    $$
#
#    Case $z = 0$ is solved in part b, objective 7.5. For $z = 1$, the labor constraint
#    becomes $x + 2y \le 15$, and the corner of the feasible region moves to where
#    $x = 5$ meets $y = 3.5$ (labor used: $5 + 7 = 12 \le 15$), giving objective
#    $5 + 3.5 - 0.25 = 8.25$. Since $8.25 > 7.5$, it is optimal to use the exchange.
# ::::

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
# d. Now the profit for the first 2 units of A is 1.5 euro per unit, and from 2 units
#    onward it drops to 0.5 euro per unit (e.g. 3.5 units of A give a profit of
#    $2 \times 1.5 + 1.5 \times 0.5 = 3.75$ euro). Model this as an LO problem and derive
#    the optimal solution.
# :::
#

# %% tags=["remove-cell"] label="hw-8-2b"
fig, ax = plt.subplots(figsize=(5, 4))
lim_x, lim_y = (0.0, 5.5), (0.0, 8.0)
draw_region(ax, [(0, 0), (3.75, 0), (2.5, 2.5), (0, 5)], lim_x, lim_y)
draw_line(ax, 1, 1, 5, lim_x, color="C0", label=r"$x + y \leq 5$ (flour)")
draw_line(ax, 2, 1, 7.5, lim_x, color="C1", label=r"$2x + y \leq 7.5$ (yeast)")
draw_line(
    ax, 1.5, 1, 6.25, lim_x, color="k", ls="--", label="$1.5x + y = 6.25$"
)
ax.plot(
    2.5, 2.5, "C3*", markersize=14, zorder=4, clip_on=False, label="optimum"
)
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% tags=["remove-cell"] label="hw-8-2d"
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
draw_region(ax_left, [(0, 0), (2, 0), (2, 3), (0, 5)], lim_x, lim_y)
draw_line(ax_left, 1, 1, 5, lim_x, color="C0", label=r"$x + y \leq 5$")
draw_line(ax_left, 2, 1, 7.5, lim_x, color="C1", label=r"$2x + y \leq 7.5$")
draw_line(ax_left, 1, 0, 2, lim_x, color="C2", label=r"$x \leq 2$")
draw_line(
    ax_left, 1.5, 1, 6, lim_x, color="k", ls="--", label="$1.5x + y = 6$"
)
ax_left.plot(
    2, 3, "C3*", markersize=14, zorder=4, clip_on=False, label="optimum"
)
ax_left.set_title("case $z = 0$")
ax_left.legend(loc="upper right", fontsize=8)

lim_y_case, lim_z_case = (0.0, 4.0), (0.0, 3.0)
draw_region(
    ax_right,
    [(0, 0), (3, 0), (2.5, 0.5), (0, 1.75)],
    lim_y_case,
    lim_z_case,
    xlabel="$y$",
    ylabel="$z$",
)
draw_line(ax_right, 1, 1, 3, lim_y_case, color="C0", label=r"$y + z \leq 3$")
draw_line(
    ax_right, 1, 2, 3.5, lim_y_case, color="C1", label=r"$y + 2z \leq 3.5$"
)
draw_line(
    ax_right,
    1,
    0.5,
    3,
    lim_y_case,
    color="k",
    ls="--",
    label="$3 + y + 0.5z = 6$",
)
ax_right.plot(
    3, 0, "C3*", markersize=14, zorder=4, clip_on=False, label="optimum"
)
ax_right.set_title("case $z > 0$, so $x = 2$")
ax_right.legend(loc="upper right", fontsize=8)
plt.show()

# %% [markdown]
#
# ::::{solution} hw-8-2
# :label: sol-hw-8-2
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of bread types A and B. Then the LO
#    is
#
#    $$
#    \begin{aligned}
#    \max \quad & 1.5x + y \\
#    \text{s.t.} \quad & x + y \le 5 \\
#    & 2x + y \le 7.5 \\
#    & x, y \ge 0.
#    \end{aligned}
#    $$
#
# b. The figure below shows the feasible region. Shifting the objective line
#    $1.5x + y = c$ up as far as possible, it last touches the feasible region where the
#    lines $x + y = 5$ and $2x + y = 7.5$ cross. Subtracting the first equation from the
#    second gives $x = 2.5$, and then $y = 2.5$: the optimal solution is
#    $(x, y) = (2.5, 2.5)$ with objective value 6.25.
#
# :::{figure} #hw-8-2b
# :label: fig-hw-8-2b
#
# The feasible region of part a, with the objective line $1.5x + y = 6.25$ through the
# optimum $(2.5, 2.5)$.
# :::
#
# c. The objective function is $p_A x + p_B y$.
#
#    i. $(p_A, p_B) = (0, 1)$: since A earns no profit, it is optimal to bake as much B as
#       possible. The flour constraint limits this to 5 units of B, so $(x, y) = (0, 5)$.
#
#    ii. The point $(1, 4)$ lies on the flour constraint line $x + y = 5$ and satisfies
#        the yeast constraint ($2 + 4 = 6 \le 7.5$), so it lies on the edge of the
#        feasible region between $(0, 5)$ and $(2.5, 2.5)$. Choosing the objective line
#        parallel to this edge, e.g. $(p_A, p_B) = (1, 1)$, makes every point on the
#        edge optimal, including $(1, 4)$.
#
# d. Let nonnegative $z$ be the less profitable version of $x$ that is unrestricted from
#    above, and restrict $x$ itself by the upper bound 2. The LO becomes
#
#    $$
#    \begin{aligned}
#    \max \quad & 1.5x + y + 0.5z \\
#    \text{s.t.} \quad & x + y + z \le 5 \\
#    & 2x + y + 2z \le 7.5 \\
#    & x \le 2 \\
#    & x, y, z \ge 0.
#    \end{aligned}
#    $$
#
#    Since $x$ and $z$ use the same resources and $x$ is more profitable, an optimal
#    solution only has $z > 0$ if $x$ is at its upper bound 2. There are two cases (see
#    the figure below):
#
#    - $z = 0$: then the model of part a applies with the extra constraint $x \le 2$.
#      Drawing this constraint in the figure of part b shows that the optimum is now
#      $(x, y, z) = (2, 3, 0)$, with objective value $1.5 \cdot 2 + 3 = 6$.
#    - $z > 0$: then $x = 2$, and the LO reduces to
#
#      $$
#      \begin{aligned}
#      \max \quad & 3 + y + 0.5z \\
#      \text{s.t.} \quad & y + z \le 3 \\
#      & y + 2z \le 3.5 \\
#      & y, z \ge 0.
#      \end{aligned}
#      $$
#
#      In the $(y, z)$ plane, the objective line reaches furthest where $y + z = 3$
#      crosses the axis $z = 0$, so the optimum is $(x, y, z) = (2, 3, 0)$ with objective
#      value 6. So a positive $z$ does not pay off.
#
#    Both cases lead to the same solution, so $(x, y, z) = (2, 3, 0)$ with profit 6 is
#    optimal: bake 2 units of A and 3 units of B.
#
#    :::{figure} #hw-8-2d
#    :label: fig-hw-8-2d
#
#    The two cases of part d. Left: $z = 0$, the region of part b with $x \le 2$ added.
#    Right: $z > 0$, so $x = 2$, and the remaining problem in $y$ and $z$.
#    :::
# ::::

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
# ::::{solution} hw-8-3
# :label: sol-hw-8-3
# :class: dropdown
#
# a. Number the items 1-4 and let $x_i = 1$ if item $i$ is taken, 0 otherwise. The ILO is
#
#    $$
#    \begin{aligned}
#    \max \quad & 3x_1 + x_2 + 8x_3 + 5x_4 \\
#    \text{s.t.} \quad & 2x_1 + 2x_2 + 4x_3 + 4x_4 \le 9 \\
#    & x_i \in \{0, 1\} \text{ for all } i.
#    \end{aligned}
#    $$
#
#    All rewards are positive, so an optimal solution leaves no room for another item. Each
#    item weighs at least 2 kg and items 3 and 4 together already weigh 8 kg, so the
#    solutions that use the capacity as much as possible are items $\{1, 2, 3\}$ (8 kg,
#    reward 12), items $\{1, 2, 4\}$ (8 kg, reward 9) and items $\{3, 4\}$ (8 kg, reward
#    13). The optimal solution is $(x_1, x_2, x_3, x_4) = (0, 0, 1, 1)$ with reward 13.
#
# b. As in
#    [Integer Optimization](lecture8_integer-optimization.ipynb#knapsack-lo-relaxation),
#    fill the knapsack in decreasing order of the reward-to-weight ratios, which are
#    $1.5, 0.5, 2, 1.25$ euro/kg for items 1-4. The order is 3, 1, 4, 2. Items 3 and 1 fit
#    completely (total weight $4 + 2 = 6$), leaving 3 kg for item 4 (weight 4), i.e. a
#    fraction $3/4$ of it. The optimal solution of the LO relaxation is
#    $(x_1, x_2, x_3, x_4) = (1, 0, 1, 3/4)$ with objective value $3 + 8 + 3.75 = 14.75$.
#
# c. We follow the
#    [branch and bound](lecture8_integer-optimization.ipynb#branch-and-bound) procedure
#    and solve every LO relaxation as in part b, packing the items fixed to 1 first. The
#    order in which subproblems are picked from the pool is a choice. Here we solve both
#    subproblems of the root first and then continue depth first: we always pick the most
#    recently created subproblem, with the branch $x_i = 0$ before $x_i = 1$.
#
#    - **Step 1: root problem.** From part b: $(1, 0, 1, 0.75)$ with value 14.75, a UB.
#      We branch on $x_4$.
#    - **Step 2: $x_4 = 0$.** Items 3, 1 and 2 all fit (8 kg): $(1, 1, 1, 0)$ with value
#      12. This solution is integer, so it gives the current best LB of 12.
#    - **Step 3: $x_4 = 1$.** Item 4 uses 4 kg; item 3 then fits and half of item 1 fills
#      the last kg: $(0.5, 0, 1, 1)$ with value 14.5. This UB is larger than 12, so we
#      branch on $x_1$.
#    - **Step 4: $x_4 = 1$, $x_1 = 0$.** Items 4 and 3 fit, and half of item 2 fills the
#      last kg: $(0, 0.5, 1, 1)$ with value 13.5, a UB larger than 12. We branch on $x_2$.
#    - **Step 5: $x_4 = 1$, $x_1 = 0$, $x_2 = 0$.** Items 4 and 3: $(0, 0, 1, 1)$ with
#      value 13. This solution is integer and improves the best LB to 13.
#    - **Step 6: suboptimal.** The solution of step 2 has value $12 < 13$.
#    - **Step 7: $x_4 = 1$, $x_1 = 0$, $x_2 = 1$.** Items 4 and 2 use 6 kg, and $3/4$ of
#      item 3 fills the rest: $(0, 1, 0.75, 1)$ with value 12, a UB.
#    - **Step 8: eliminate.** The UB of 12 is below the best LB of 13.
#    - **Step 9: $x_4 = 1$, $x_1 = 1$.** Items 4 and 1 use 6 kg, and $3/4$ of item 3 fills
#      the rest: $(1, 0, 0.75, 1)$ with value 14, a UB larger than 13. We branch on $x_3$.
#    - **Step 10: $x_4 = 1$, $x_1 = 1$, $x_3 = 0$.** Items 4 and 1, and item 2 still fits:
#      $(1, 1, 0, 1)$ with value 9, an integer solution and thus a LB.
#    - **Step 11: suboptimal.** The LB of 9 is below the best LB of 13.
#    - **Step 12: $x_4 = 1$, $x_1 = 1$, $x_3 = 1$.** The items fixed to 1 weigh
#      $4 + 2 + 4 = 10 > 9$ kg, so this subproblem is infeasible and is eliminated.
#    - **Step 13: optimum.** The pool is empty, so the best solution found,
#      $(0, 0, 1, 1)$ with reward 13 from step 5, is optimal. This matches part a.
#
# :::{figure} images/lecture8_hw1-ex3c.png
# :label: fig-hw-8-3c
#
# Branch-and-bound tree for part c (UB = upper bound from the LO relaxation, LB = lower
# bound from a feasible integer solution). The infeasible subproblem of step 12 has no
# solution; its second line shows the values fixed by branching.
# :::
#
# d. Add the constraint $4x_1 + 8x_2 + 2x_3 + 10x_4 \le 12$ to the model of part a. An
#    extra constraint can only make the feasible region smaller, so the optimal value
#    cannot increase. The solution $(0, 0, 1, 1)$ from part c uses volume
#    $2 + 10 = 12 \le 12$, so it is still feasible, and with reward 13 it is still
#    optimal.
# ::::

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

# %% tags=["remove-cell"] label="hw-8-4b"
fig, ax = plt.subplots(figsize=(5, 4))
lim_x, lim_y = (0.0, 4.5), (0.0, 5.0)
draw_region(ax, [(0, 0), (3, 0), (16 / 7, 20 / 7), (0, 4)], lim_x, lim_y)
draw_line(ax, 4, 1, 12, lim_x, color="C0", label=r"$4x + y \leq 12$")
draw_line(ax, 2, 4, 16, lim_x, color="C1", label=r"$2x + 4y \leq 16$")
draw_line(ax, 1, 1, 36 / 7, lim_x, color="k", ls="--", label="$x + y = 36/7$")
int_points = [
    (i, j)
    for i in range(5)
    for j in range(6)
    if 4 * i + j <= 12 and 2 * i + 4 * j <= 16
]
ax.scatter(
    [i for i, _ in int_points],
    [j for _, j in int_points],
    color="C2",
    zorder=3,
    label="integer feasible points",
    clip_on=False,
)
ax.plot(
    16 / 7,
    20 / 7,
    "C3*",
    markersize=14,
    zorder=4,
    clip_on=False,
    label="LO optimum",
)
ax.plot(
    2, 3, "ko", markersize=8, zorder=4, clip_on=False, label="integer optimum"
)
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% [markdown]
#
# ::::{solution} hw-8-4
# :label: sol-hw-8-4
# :class: dropdown
#
# a. Let $x$ and $y$ be the number of units produced of product 1 and 2. Then
#
#    $$
#    \begin{aligned}
#    \max \quad & x + y \\
#    \text{s.t.} \quad & 4x + y \le 12 \\
#    & 2x + 4y \le 16 \\
#    & x, y \ge 0.
#    \end{aligned}
#    $$
#
# b. The figure below shows the feasible region. Shifting the objective line $x + y = c$
#    up as far as possible, it last touches the feasible region where the lines
#    $4x + y = 12$ and $2x + 4y = 16$ cross. From the first line, $y = 12 - 4x$;
#    substituting in the second gives $2x + 48 - 16x = 16$, so $x = 16/7$ and
#    $y = 20/7$, with objective value $36/7 \approx 5.14$.
#
# :::{figure} #hw-8-4b
# :label: fig-hw-8-4b
#
# The feasible region of part a, with the objective line through the LO optimum
# $(16/7, 20/7)$ and the feasible integer points used in part c.
# :::
#
# c. Add the constraints $x, y \in \{0, 1, 2, \dots\}$. The LO optimum of part b is an
#    upper bound: no integer solution has a value above $36/7 \approx 5.14$. With $x$ and
#    $y$ integer, the value $x + y$ is integer too, so it is at most 5. The point
#    $(x, y) = (2, 3)$ is feasible ($8 + 3 = 11 \le 12$ and $4 + 12 = 16 \le 16$) and
#    has value 5, so it reaches this bound and is optimal. In the figure: shifting the
#    objective line down from the LO optimum, $(2, 3)$ is the first feasible integer
#    point it reaches.
#
# d. Constraint $2x + 4y \le 16$ becomes $2x + 4y \le 20$. The LO optimum is now where
#    $4x + y = 12$ meets $2x + 4y = 20$: $x = 2$, $y = 4$, objective value 6. Since this
#    solution is already integer, it is also optimal for the integer case. The additional
#    revenue compared to part c is $6 - 5 = 1$.
# ::::

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
# b. Rewrite it in the
#    [general form](lecture8_integer-optimization.ipynb#general-formulation)
#    (maximization, only "$\le$" constraints).
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
