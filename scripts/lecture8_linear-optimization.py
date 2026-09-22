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
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_linear-optimization.ipynb)

# %% [markdown]
# This notebook introduces linear optimization (LO), also called linear programming. It is
# a widely used optimization framework in practice: efficient solvers exist that are
# guaranteed to find the optimum even for problems with thousands of variables and
# constraints. We introduce it through an example, model it, solve it in Python with
# [pulp](https://coin-or.github.io/pulp/), look at what the solver is doing both graphically
# and algebraically, and see what can go wrong.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate a real-life decision problem as an LO model (decision variables, objective,
#   constraints);
# - implement an LO model in pulp, both directly and with the data kept separate from the
#   model;
# - understand the intuition behind the simplex method;
# - interpret an LO solution both graphically and algebraically;
# - recognize infeasibility and unboundedness in pulp's solve status.

# %% [markdown]
# ## A Motivating Problem: Optimal Product Mix
#
# A workshop makes two products from two limited resources. Producing one unit of product
# $x$ uses 1 unit of resource 1 and 1 unit of resource 2; one unit of product $y$ uses 0
# units of resource 1 and 2 units of resource 2. Product $x$ earns a profit of 2 per unit,
# product $y$ a profit of 3 per unit.
#
# | | profit | resource 1 needed | resource 2 needed |
# |---|---|---|---|
# | $x$ | 2 | 1 | 1 |
# | $y$ | 3 | 0 | 2 |
#
# There are 5 units of resource 1 and 10 units of resource 2 available. Which product mix
# maximizes profit?

# %% [markdown]
# ## The Modeling Approach
#
# We follow the same four steps for every problem in this course:
#
# 1. **study the problem** in detail (done above);
# 2. **define the decision variables**: let $x$ and $y$ be the quantities of the two
#    products produced;
# 3. **define the objective**: maximize profit, $2x + 3y$;
# 4. **define the constraints**: the resource-1 usage, $x$, cannot exceed 5; the resource-2
#    usage, $x + 2y$, cannot exceed 10; and $x, y \ge 0$.
#
# In mathematical form:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x + 3y & \text{(objective)} \\
# \text{subject to} \quad & x \le 5 & \text{(resource 1)} \\
# & x + 2y \le 10 & \text{(resource 2)} \\
# & x, y \ge 0.
# \end{aligned}
# $$
#
# The objective and all constraints are linear functions of the decision variables
# $(x, y)$, hence the name *linear* optimization.

# %% [markdown]
# ## Solving the Model Directly in pulp
#
# For a small, one-off model like this, the fastest way from model to answer is to write it
# in pulp exactly as it appears above, with the numbers typed directly into the variable
# bounds and constraints:

# %%
import matplotlib.pyplot as plt
import numpy as np

try:
    import pulp
except ModuleNotFoundError:  # pulp is not preinstalled on Google Colab
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pulp"], check=True)
    import pulp

# %%
product_mix = pulp.LpProblem(name="product_mix", sense=pulp.LpMaximize)
x = pulp.LpVariable(name="x", lowBound=0)
y = pulp.LpVariable(name="y", lowBound=0)

product_mix += 2 * x + 3 * y, "profit"
product_mix += x <= 5, "resource_1"
product_mix += x + 2 * y <= 10, "resource_2"

product_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[product_mix.status])
print(f"x = {x.value()}, y = {y.value()}")
print("optimal profit:", product_mix.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-1
#
# Derive by hand the corner points of the feasible region of the product-mix problem above,
# evaluate the objective at each, and check which one pulp found.
# :::

# %% [markdown]
# ## Separating Data from the Model
#
# The direct version above hardcodes every number into the model itself, which is fine for
# a problem you solve once. It breaks down as soon as you want to re-solve for different
# data (more products, different resource limits) or reuse the same model code elsewhere:
# every number would have to be found and edited inside the model-building code, which is
# slow and error-prone. The fix is to keep the instance *data* in plain Python structures,
# separate from the *model*-building code, and build the model from that data:

# %%
profit = {"x": 2, "y": 3}
resource_use = {  # resource_use[r][p]: units of resource r per unit of product p
    "resource 1": {"x": 1, "y": 0},
    "resource 2": {"x": 1, "y": 2},
}
available = {"resource 1": 5, "resource 2": 10}
products = list(profit)

product_mix = pulp.LpProblem(name="product_mix", sense=pulp.LpMaximize)
q = {p: pulp.LpVariable(name=p, lowBound=0) for p in products}

product_mix += pulp.lpSum(profit[p] * q[p] for p in products), "profit"
for r, cap in available.items():
    product_mix += (
        pulp.lpSum(resource_use[r][p] * q[p] for p in products) <= cap,
        r.replace(" ", "_"),
    )

product_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[product_mix.status])
print("optimal mix:", {p: q[p].value() for p in products})
print("optimal profit:", product_mix.objective.value())

# %% [markdown]
# The model-building code (the `for` loop and the `pulp.lpSum` calls) no longer mentions 2,
# 3, 5, or 10 anywhere: it works unchanged for any number of products and resources, as
# long as `profit`, `resource_use`, and `available` describe them. This is the pattern we
# use for every larger model from here on, and it is also the discipline behind the
# algebraic modeling languages (AMLs) introduced in
# [Modeling Tools and Solvers](lecture10_modeling-tools.ipynb).

# %% [markdown]
# ## A Larger Example in pulp
#
# The same three ingredients, and the same data/model split, scale to any size. Consider a
# three-product version with two resources:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x_1 + 4x_2 + 8x_3 \\
# \text{subject to} \quad & x_1 + 3x_2 + 2x_3 \le 10 \\
# & x_1 + 3x_3 \le 12 \\
# & x_1, x_2, x_3 \ge 0.
# \end{aligned}
# $$
#
# The coefficients naturally form a matrix (one row per constraint, one column per
# variable), which we keep as a plain nested list:

# %%
objective_coefs = [2, 4, 8]
constraint_coefs = [[1, 3, 2], [1, 0, 3]]
rhs = [10, 12]

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
# The model-building code is identical in structure to the two-variable case; only the data
# changed.
#
# :::{exercise}
# :label: ex-6-4
#
# Re-solve the larger LO problem above adding the two constraints one at a time, and note
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
# a. Next to the 10 units of the first resource, you can buy extra units of it for a price
#    of 1 per unit. What is the optimal solution now?
#
# b. The same, but the price per extra unit is 3.
#
# c. The same, but the price is 6. Can you interpret the result?
# :::

# %% [markdown]
# ## A Graphical View
#
# With only two decision variables we can draw the product-mix problem. Each constraint is
# a line; the feasible region is the set of points satisfying all of them at once. Because
# both the constraints and the objective are linear, the objective-value contour lines are
# straight and parallel. Sliding one in the direction of increasing profit, the last
# feasible point it touches is an optimal solution, and it is always a corner of the
# feasible region.
#
# The code below only produces the figure; reproducing it is not itself something you need
# to learn, so feel free to skip straight to the plot.

# %% tags=["hide-input"]
x_grid = np.linspace(0, 6, 200)
plt.figure(figsize=(5, 5))
plt.axvline(5, color="C0", label=r"$x \leq 5$ (resource 1)")
plt.plot(x_grid, (10 - x_grid) / 2, color="C1", label=r"$x + 2y \leq 10$ (resource 2)")
plt.fill_between(
    x_grid, 0, (10 - x_grid) / 2, where=(x_grid <= 5), alpha=0.2, label="feasible region"
)

x_opt, y_opt = q["x"].value(), q["y"].value()
for level, style in [(16, ":"), (product_mix.objective.value(), "-")]:
    plt.plot(x_grid, (level - 2 * x_grid) / 3, style, color="grey")
plt.plot(x_opt, y_opt, "ko")
plt.annotate(
    f"optimum ({x_opt:.1f}, {y_opt:.1f})", (x_opt, y_opt), textcoords="offset points", xytext=(8, 8)
)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="upper right", fontsize=8)
plt.title("The two grey lines are objective contours; the solid one is optimal")
plt.show()

# %% [markdown]
# The dotted grey line (profit 16) still has feasible points but we can do better; sliding
# it out until it just leaves the feasible region gives the solid line, which touches the
# region only at the corner where the two resource constraints meet: the optimum pulp
# reported above.

# %% [markdown]
# ## An Algebraic View
#
# The same optimum can be understood without a picture. Turn each "$\le$" constraint into
# an equality by adding a non-negative slack variable, the amount of the resource left
# unused:
#
# $$
# \begin{aligned}
# x \le 5 \\
# x + 2y \le 10 \\
# x, y \ge 0
# \end{aligned}
# \quad\Leftrightarrow\quad
# \begin{aligned}
# x + s_1 &= 5 \\
# x + 2y + s_2 &= 10 \\
# x, y, s_1, s_2 &\ge 0
# \end{aligned}
# $$
#
# There are now 2 equations and 4 variables. As a rule, a system of $k$ independent linear
# equations in $k$ unknowns has exactly one solution, so if we set any 2 of the 4 variables
# to zero the other 2 are determined. Each such choice corresponds to a corner of the
# feasible region (some combinations give a negative value and are infeasible; those are
# corners of the lines' intersections that lie outside the region). The corners here are:
#
# - $(x, y, s_1, s_2) = (0, 0, 5, 10)$: the origin;
# - $(5, 0, 0, 5)$: resource 1 fully used;
# - $(0, 5, 5, 0)$: resource 2 fully used;
# - $(5, 2.5, 0, 0)$: both resource constraints tight, the optimum.
#
# The simplex algorithm (G.B. Dantzig, 1947), which pulp's default solver uses for LO, hops
# from corner to neighbouring corner, each time to one with a better objective value, and
# stops when no neighbouring corner is better. Why is that enough to guarantee the *global*
# optimum? Because the feasible region of an LO problem is a convex polyhedron: a shape with
# flat faces and no hidden hills. If every neighbour of your current corner is worse, there
# is nowhere higher to go, since reaching a higher point would require the boundary to
# curve, and linear constraints never curve. So for LO, a local optimum is automatically a
# global optimum.
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
# 2. **unbounded**: solutions of arbitrarily large objective value exist. For the
#    product-mix problem this happens if we drop the resource-2 constraint, since we could
#    then make unlimited units of $y$;
# 3. **infeasible**: no point satisfies all constraints. This happens if, for example, a
#    customer contract forces $y \ge 12$ while the resource-2 constraint allows at most
#    $y = 5$.
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
# The same two problems, this time solved with pulp:

# %%
unbounded_lp = pulp.LpProblem(name="unbounded_example", sense=pulp.LpMaximize)
x = pulp.LpVariable(name="x", lowBound=0)
y = pulp.LpVariable(name="y", lowBound=0)
unbounded_lp += 2 * x + 3 * y
unbounded_lp += x <= 5  # the resource-2 constraint is dropped

unbounded_lp.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[unbounded_lp.status])

# %%
infeasible_lp = pulp.LpProblem(name="infeasible_example", sense=pulp.LpMaximize)
x = pulp.LpVariable(name="x", lowBound=0)
y = pulp.LpVariable(name="y", lowBound=0)
infeasible_lp += 2 * x + 3 * y
infeasible_lp += x <= 5
infeasible_lp += x + 2 * y <= 10
infeasible_lp += y >= 12  # the customer contract

infeasible_lp.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[infeasible_lp.status])

# %% [markdown]
# :::{exercise}
# :label: ex-6-3
#
# Formulate and solve, in pulp, one more unbounded and one more infeasible variant of the
# product-mix problem of your own, and check that `LpStatus` reports what you expect.
# :::

# %% [markdown]
# (project-planning)=
# ## Another Application: Project Planning
#
# (graphs-intro)=
# A graph is a network of *nodes* (also called vertices) connected by *edges*; a directed
# edge is called an *arc*. Many problems are naturally expressed on graphs. One is project
# planning: a project has activities (nodes), each with a duration, and precedence
# relations (arcs). An activity can only start once its predecessors are finished (you
# cannot roof a house before the walls are up).
#
# | activity | duration | must follow |
# |---|---|---|
# | A | 3 | none |
# | B | 4 | none |
# | C | 2 | A |
# | D | 5 | A, B |
# | E | 1 | C, D |
# | F | 3 | D |
#
# Let $x_i$ be the finish time of activity $i$. If $i$ precedes $j$ then
# $x_i + d_j \le x_j$, where $d_j$ is the duration of $j$; also $x_i \ge d_i$. We want the
# time when *all* activities are finished, $\max_i x_i$. That maximum is not linear, but we
# can introduce a variable $z \ge x_i$ for all $i$ and minimize $z$: the optimization will
# push $z$ down to exactly the largest finish time. This "min-max" trick reappears
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
# found by a direct algorithm without LO, and that algorithm extends to *random* durations,
# which matters because durations are often hard to predict, one reason IT projects
# overrun.
#
# :::{exercise}
# :label: ex-6-7
#
# Find the project finish time by hand: repeatedly take the activity whose predecessors are
# all finished and give it the earliest possible finish time. Check you get the same
# makespan as pulp.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.1 "Problem Formulation",
#   §6.3 "Example LO Problems" (project planning).
