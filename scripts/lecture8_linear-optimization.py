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
# # Lecture 8: Linear Optimization

# %% [markdown]
# This notebook introduces **linear optimization (LO)** — also called linear programming.
# It is the most widely used optimization framework in practice: efficient solvers exist
# that are guaranteed to find the optimum even for problems with thousands of variables and
# constraints. We introduce it through an example, look at what the solver is doing both
# graphically and algebraically, see what can go wrong, and end with the general
# formulation. We solve everything in Python with
# [pulp](https://coin-or.github.io/pulp/).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pulp

# %% [markdown]
# (product-mix)=
# ## A Motivating Problem: Optimal Product Mix
#
# Recall from [the introduction](lecture8_introduction.ipynb) that a real-life optimization
# problem has three ingredients: a **decision**, a **system** that limits the allowed
# decisions, and an **outcome** to be optimized.
#
# A workshop makes two products, **bookcases** and **desks**, from two limited resources,
# **oak panels** and **assembly hours**:
#
# | | profit (€) | oak panels needed | assembly hours needed |
# |---|---|---|---|
# | bookcase | 3 | 1 | 2 |
# | desk     | 5 | 3 | 1 |
#
# There are 12 oak panels and 10 assembly hours available this week. Which product mix
# maximizes profit?
#
# (Excel and other spreadsheet tools can also solve small problems like this — for example
# via `SUMPRODUCT` plus the Solver add-in — but this course works in Python throughout.)

# %% [markdown]
# ### The Modeling Approach
#
# We follow the same four steps for every problem:
#
# 1. **study the problem** in detail (done above);
# 2. **define the decision variables**: let $b$ and $d$ be the number of bookcases and
#    desks produced;
# 3. **define the objective**: maximize profit, $3b + 5d$;
# 4. **define the constraints**: the oak panels used, $b + 3d$, cannot exceed 12; the
#    assembly hours used, $2b + d$, cannot exceed 10; and $b, d \ge 0$.
#
# In mathematical form:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 3b + 5d & \text{(objective)} \\
# \text{subject to} \quad & b + 3d \le 12 & \text{(oak panels)} \\
# & 2b + d \le 10 & \text{(assembly hours)} \\
# & b, d \ge 0.
# \end{aligned}
# $$
#
# The objective and all constraints are **linear** functions of the decision variables
# $(b, d)$ — hence *linear* optimization.
#
# To solve it with pulp we keep the problem **data** separate from the **model**:

# %%
profit = {"bookcase": 3, "desk": 5}
resource_use = {  # resource_use[r][p]: units of resource r per unit of product p
    "oak panels": {"bookcase": 1, "desk": 3},
    "assembly hours": {"bookcase": 2, "desk": 1},
}
available = {"oak panels": 12, "assembly hours": 10}
products = list(profit)

# %% [markdown]
# The model is an `LpProblem`, one non-negative continuous variable per product, the
# objective, and one constraint per resource:

# %%
mix = pulp.LpProblem(name="product_mix", sense=pulp.LpMaximize)
x = {p: pulp.LpVariable(name=p.replace(" ", "_"), lowBound=0) for p in products}

mix += pulp.lpSum(profit[p] * x[p] for p in products), "profit"
for r, cap in available.items():
    mix += pulp.lpSum(resource_use[r][p] * x[p] for p in products) <= cap, r.replace(" ", "_")

mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[mix.status])
print("optimal mix:", {p: x[p].value() for p in products})
print("optimal profit:", mix.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-1
#
# Derive by hand the corner points of the feasible region of the product-mix problem above,
# evaluate the objective at each, and check which one pulp found.
# :::

# %% [markdown]
# ## A Graphical View
#
# With only two decision variables we can draw the problem. Each constraint is a line; the
# feasible region is the set of points satisfying all of them at once. Because both the
# constraints and the objective are linear, the objective-value contour lines are straight
# and parallel, and sliding one in the direction of increasing profit, the last feasible
# point it touches is an optimal solution — and it is always a **corner** of the feasible
# region.

# %%
b = np.linspace(0, 6, 200)
plt.figure(figsize=(5, 5))
plt.plot(b, (12 - b) / 3, label=r"$b + 3d \leq 12$ (oak panels)")
plt.plot(b, 10 - 2 * b, label=r"$2b + d \leq 10$ (assembly hours)")

# feasible region: below both lines, first quadrant
d_upper = np.minimum((12 - b) / 3, 10 - 2 * b)
plt.fill_between(b, 0, d_upper, where=d_upper > 0, alpha=0.2, label="feasible region")

b_opt, d_opt = x["bookcase"].value(), x["desk"].value()
for level, style in [(15, ":"), (mix.objective.value(), "-")]:
    plt.plot(b, (level - 3 * b) / 5, style, color="grey")
plt.plot(b_opt, d_opt, "ko")
plt.annotate(
    f"optimum ({b_opt:.1f}, {d_opt:.1f})", (b_opt, d_opt), textcoords="offset points", xytext=(8, 8)
)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel("bookcases $b$")
plt.ylabel("desks $d$")
plt.legend(loc="upper right", fontsize=8)
plt.title("The two grey lines are objective contours; the solid one is optimal")
plt.show()

# %% [markdown]
# The dotted grey line (profit 15) still has feasible points but we can do better; sliding
# it out until it just leaves the feasible region gives the solid line, which touches the
# region only at the corner where the two resource constraints meet — the optimum pulp
# reported above.

# %% [markdown]
# ## An Algebraic View
#
# The same optimum can be understood without a picture. Turn each "$\le$" constraint into
# an equality by adding a non-negative **slack variable** — the amount of the resource left
# unused:
#
# $$
# \begin{aligned}
# b + 3d \le 12 \\
# 2b + d \le 10 \\
# b, d \ge 0
# \end{aligned}
# \quad\Leftrightarrow\quad
# \begin{aligned}
# b + 3d + s_1 &= 12 \\
# 2b + d + s_2 &= 10 \\
# b, d, s_1, s_2 &\ge 0
# \end{aligned}
# $$
#
# There are now 2 equations and 4 variables. As a rule, a system of $k$ independent linear
# equations in $k$ unknowns has exactly one solution, so if we set any 2 of the 4 variables
# to zero the other 2 are determined. Each such choice corresponds to a **corner** of the
# feasible region (some combinations give a negative value and are infeasible; those are
# corners of the lines' intersections that lie outside the region). The corners here are:
#
# - $(b, d, s_1, s_2) = (0, 0, 12, 10)$ — the origin;
# - $(0, 4, 0, 6)$ — oak panels fully used;
# - $(5, 0, 7, 0)$ — assembly hours fully used;
# - the point where both resource constraints are tight ($s_1 = s_2 = 0$) — the optimum.
#
# The **simplex algorithm** (G.B. Dantzig, 1947), which pulp's default solver uses for LO,
# hops from corner to neighbouring corner, each time to one with a better objective value,
# and stops when no neighbouring corner is better. Why is that enough to guarantee the
# *global* optimum? Because the feasible region of an LO problem is a convex polyhedron — a
# shape with flat faces and no "hidden hills". If every neighbour of your current corner is
# worse, there is nowhere higher to go: reaching a higher point would require the boundary
# to curve, and linear constraints never curve. So for LO, a local optimum is automatically
# a global optimum.
#
# :::{note} History of Linear Optimization
# Several researchers formulated linear optimization problems, but it was G.B. Dantzig
# (1914–2005) who invented the simplex algorithm in 1947. Until recently the field was
# commonly called *linear programming*. Its dynamic, random counterpart is *dynamic
# programming*, developed by R.E. Bellman (1920–1984); the division between deterministic
# and random problems is still very visible in operations research.
# :::

# %% [markdown]
# ## Three Possible Outcomes
#
# Not every LO problem has an optimal solution. There are exactly three possibilities:
#
# 1. **an optimal solution exists** (the case above);
# 2. **unbounded** — solutions of arbitrarily large objective value exist. For the
#    product-mix problem this happens if we drop the assembly-hours constraint: we could
#    then make unlimited desks;
# 3. **infeasible** — no point satisfies all constraints. This happens if, for example, a
#    customer contract forces $d \ge 12$ while the oak-panel constraint allows at most
#    $d = 4$.
#
# pulp reports these as the `LpStatus` values `"Unbounded"` and `"Infeasible"` instead of
# `"Optimal"`. [](#fig-lo-degenerate) shows both situations schematically.

# %% [markdown]
# :::{figure} images/lecture8_fig6.2.png
# :label: fig-lo-degenerate
#
# An unbounded (left) and an infeasible (right) LO problem.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-3
#
# Enter the unbounded and the infeasible variants described above in pulp and check what
# `LpStatus` you get for each.
# :::

# %% [markdown]
# ## A Larger Example in pulp
#
# The same three ingredients scale to any size. Consider:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 5x_1 + 4x_2 + 3x_3 \\
# \text{subject to} \quad & 2x_1 + 3x_2 + x_3 \le 5 \\
# & 4x_1 + x_2 + 2x_3 \le 11 \\
# & 3x_1 + 4x_2 + 2x_3 \le 8 \\
# & x_1, x_2, x_3 \ge 0.
# \end{aligned}
# $$
#
# The coefficients naturally form a matrix (one row per constraint, one column per
# variable), which we keep as a plain nested list:

# %%
objective_coefs = [5, 4, 3]
constraint_coefs = [[2, 3, 1], [4, 1, 2], [3, 4, 2]]
rhs = [5, 11, 8]

n_vars = len(objective_coefs)

larger_lo = pulp.LpProblem(name="larger_lo", sense=pulp.LpMaximize)
z = [pulp.LpVariable(name=f"x_{k + 1}", lowBound=0) for k in range(n_vars)]

larger_lo += pulp.lpSum(objective_coefs[k] * z[k] for k in range(n_vars))
for row, bound in zip(constraint_coefs, rhs):
    larger_lo += pulp.lpSum(row[k] * z[k] for k in range(n_vars)) <= bound

larger_lo.solve(pulp.PULP_CBC_CMD(msg=False))
print("optimal solution:", [zk.value() for zk in z])
print("optimal objective:", larger_lo.objective.value())

# %% [markdown]
# The model-building code is identical in structure to the two-variable case — only the
# data changed.
#
# :::{exercise}
# :label: ex-6-4
#
# Re-solve the larger LO problem above adding the three constraints one at a time, and note
# how the optimal objective value changes after each addition.
# :::
#
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
#
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
#
# :::{exercise}
# :label: ex-6-6
#
# Extend the larger LO problem above. It helps to ask what the additional *decision* is.
#
# a. Next to the 5 units of the first resource, you can buy extra units of it for a price of
#    1 per unit. What is the optimal solution now?
#
# b. The same, but the price per extra unit is 3.
#
# c. The same, but the price is 6. Can you interpret the result?
# :::

# %% [markdown]
# (project-planning)=
# ## Another Application: Project Planning
#
# (graphs-intro)=
# A **graph** is a network of *nodes* (also called vertices) connected by *edges*; a
# directed edge is called an *arc*. Many problems are naturally expressed on graphs. One is project planning: a
# project has activities (nodes), each with a duration, and precedence relations (arcs) —
# an activity can only start once its predecessors are finished (you cannot roof a house
# before the walls are up).
#
# | activity | duration | must follow |
# |---|---|---|
# | A | 3 | — |
# | B | 4 | — |
# | C | 2 | A |
# | D | 5 | A, B |
# | E | 1 | C, D |
# | F | 3 | D |
#
# Let $x_i$ be the **finish time** of activity $i$. If $i$ precedes $j$ then
# $x_i + d_j \le x_j$, where $d_j$ is the duration of $j$; also $x_i \ge d_i$. We want the
# time when *all* activities are finished, $\max_i x_i$. That maximum is not linear, but we
# can introduce a variable $z \ge x_i$ for all $i$ and **minimize** $z$ — the optimization
# will push $z$ down to exactly the largest finish time. This "min-max" trick reappears
# throughout the course.
#
# $$
# \begin{aligned}
# \text{minimize} \quad & z \\
# \text{subject to} \quad & z \ge x_i \text{ for all activities } i \\
# & x_i + d_j \le x_j \text{ if } i \text{ precedes } j \\
# & x_i \ge d_i \text{ for all activities } i.
# \end{aligned}
# $$

# %%
duration = {"A": 3, "B": 4, "C": 2, "D": 5, "E": 1, "F": 3}
precedences = [("A", "C"), ("A", "D"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F")]

# %%
project = pulp.LpProblem(name="project_planning", sense=pulp.LpMinimize)
finish = {a: pulp.LpVariable(name=f"x_{a}", lowBound=duration[a]) for a in duration}
makespan = pulp.LpVariable(name="z", lowBound=0)

for a in duration:
    project += makespan >= finish[a]
for before, after in precedences:
    project += finish[before] + duration[after] <= finish[after]

# minimize z; the tiny extra term only breaks ties, so activities that have slack still get
# their *earliest* finish time reported instead of an arbitrary one consistent with z.
project += makespan + 1e-4 * pulp.lpSum(finish[a] for a in duration)

project.solve(pulp.PULP_CBC_CMD(msg=False))
print("earliest finish times:", {a: round(finish[a].value(), 2) for a in duration})
print("project finish time (makespan):", makespan.value())

# %% [markdown]
# The project-planning model above (activities `A`–`F`, the `duration` dict and
# `precedences` list) is referred to from later notebooks. The finish time can also be
# found by a direct algorithm without LO, and that algorithm extends to *random* durations
# — relevant because durations are often hard to predict, one reason IT projects overrun.
#
# :::{exercise}
# :label: ex-6-7
#
# Find the project finish time by hand: repeatedly take the activity whose predecessors are
# all finished and give it the earliest possible finish time. Check you get the same
# makespan as pulp.
# :::

# %% [markdown]
# ## The General Formulation and Linearity
#
# In general, an LO problem with $n$ decision variables and $m$ constraints is
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{j=1}^{n} p_j x_j \\
# \text{subject to} \quad & \sum_{j=1}^{n} a_{ij} x_j \le b_i, \quad i = 1, \dots, m \\
# & x_1, \dots, x_n \ge 0,
# \end{aligned}
# $$
#
# or in matrix notation $\max\{p^T x \mid A x \le b,\ x \ge 0\}$, with $p, x$ column vectors
# of length $n$, $b$ a column vector of length $m$, and $A$ an $m \times n$ matrix.
#
# This one form covers more than it seems. Every other case can be rewritten into it:
#
# - **minimization**: $\min p^T x = -\max (-p^T x)$;
# - **"$\ge$" constraints**: $Ax \ge b \Leftrightarrow -Ax \le -b$;
# - **"$=$" constraints**: $Ax = b \Leftrightarrow Ax \le b$ and $Ax \ge b$;
# - **free (unrestricted) variables**: replace $x$ by $x^+ - x^-$ with $x^+, x^- \ge 0$.
#
# What *cannot* be relaxed is linearity. If the objective is nonlinear, the optimum need
# not lie at a corner (think of $\max -x^2$ on $[-1, 1]$, optimal at the interior point 0).
# If a constraint is nonlinear, the feasible region is no longer a convex polyhedron, so it
# can have several local optima and the simplex reasoning above breaks down — you are not
# sure you have found the best solution until you have checked every local optimum, which is
# usually intractable. Nonlinear optimization therefore needs different, less efficient
# algorithms. One important structured case that we *can* handle well is when variables are
# required to be integer — **integer linear optimization**, the subject of
# [the next notebook](lecture8_integer-optimization.ipynb).

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.1 "Problem Formulation",
#   §6.3 "Example LO Problems" (project planning). The book's spreadsheet material (§6.2) is
#   replaced with pulp throughout.
# - PuLP documentation: https://coin-or.github.io/pulp/
