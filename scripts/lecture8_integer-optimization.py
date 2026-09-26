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
# description: "Integer linear optimization: why integer problems are harder, how branch and bound solves them, the knapsack problem, and the general form of LO problems."
# thumbnail: null
# ---
# # Lecture 8: Integer Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_integer-optimization.ipynb)

# %% [markdown]
# An integer linear optimization (ILO) problem is an LO problem with the extra requirement
# that some or all decision variables take integer values, $x_i \in \{0, 1, 2, \dots\}$, or
# are binary, $x_i \in \{0, 1\}$. (A binary variable is just an integer one with the added
# constraint $x_i \le 1$.)
#
# In pulp this is a one-word change: set a variable's `cat` to `"Integer"` or `"Binary"`
# instead of the default `"Continuous"`. Everything else about building and solving the
# model is the same. Solving it, however, is a different matter: in general ILO is much
# harder than LO, as this notebook and [Complexity](lecture11_complexity.ipynb) explain.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize when a problem needs integer or binary decision variables, and formulate and
#   solve it in pulp;
# - explain why ILO is generally harder to solve than LO;
# - explain how branch and bound finds the optimum of an ILO problem;
# - write any LO problem in the general matrix form;
# - explain why linearity (but not integrality) is essential for efficient solvability, and
#   how integer variables let you model many other nonlinearities without giving that up.

# %% [markdown]
# ## Why Integer Problems Are Harder
#
# Take the bookcase/desk product-mix problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb) and require whole units of $x$
# and $y$. Its LO optimum was $(x, y) = (3.6, 2.8)$ with profit 24.8, not integer. Two
# things go wrong compared to LO:
#
# - **the optimal corner is no longer feasible**, so the simplex reasoning ("the optimum is
#   at a corner") does not directly help;
# - **rounding the LO optimum is not enough**: rounding both up to $(4, 3)$ violates the
#   oak-panel constraint ($4 + 9 = 13 > 12$), and rounding both down to $(3, 2)$ is feasible
#   but only reaches a profit of 19, well below the true integer optimum. Evaluating nearby
#   integer points tells us little, because finding the true optimum in general needs a
#   systematic search.
#
# Let pulp solve the integer version:

# %%
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pulp
from matplotlib.patches import Rectangle

profit = {"bookcase": 3, "desk": 5}
# resource_use[res][p]: units of resource res per unit of product p
resource_use = {
    "oak panels": {"bookcase": 1, "desk": 3},
    "assembly hours": {"bookcase": 2, "desk": 1},
}
available = {"oak panels": 12, "assembly hours": 10}
products = list(profit)

int_mix = pulp.LpProblem(name="integer_product_mix", sense=pulp.LpMaximize)
dec_vars = {
    p: pulp.LpVariable(name=p, lowBound=0, cat="Integer") for p in products
}
int_mix += pulp.lpSum(profit[p] * dec_vars[p] for p in products)
for res, cap in available.items():
    usage = pulp.lpSum(resource_use[res][p] * dec_vars[p] for p in products)
    int_mix += usage <= cap, res.replace(" ", "_")
int_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("integer optimum:", {p: dec_vars[p].value() for p in products})
print("optimal profit:", int_mix.objective.value())

# %% [markdown]
# The picture below shows why rounding is unreliable: the LP-relaxation optimum (the star)
# does not sit on the integer grid, and the nearest lattice points are not necessarily
# feasible or optimal. The ILO optimum (the diamond) is the best *feasible* grid point,
# which can be several steps away from the naive rounding of the relaxation.

# %% tags=["remove-cell"] label="lattice-plot"
# Plot style for the light and the dark site theme: transparent background;
# on the site, custom.css gives the text and grid lines the page colours.
# TEXT_COLOR is the fallback elsewhere (Colab, local Jupyter). Zooming and
# panning are off, so a swipe over a figure scrolls the page on a phone.
GREY = "#888888"
TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "dragmode": False,
    "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.18,
    },
    "hoverlabel": {"font": {"size": 13}},
}
AXIS_STYLE = {
    "gridcolor": "rgba(128,128,128,0.25)",
    "zerolinecolor": "rgba(128,128,128,0.5)",
    "fixedrange": True,
}
PLOT_CONFIG = {"displayModeBar": False}

lattice = go.Figure()
lattice.add_trace(
    go.Scatter(
        x=[0, 5, 3.6, 0],
        y=[0, 0, 2.8, 4],
        fill="toself",
        fillcolor="rgba(44,160,44,0.18)",
        mode="none",
        name="feasible region",
        hoverinfo="skip",
    )
)
lattice.add_trace(
    go.Scatter(
        x=[0, 6],
        y=[4, 2],
        mode="lines",
        line={"color": "#1f77b4"},
        name="x + 3y ≤ 12 (oak panels)",
        hoverinfo="skip",
    )
)
lattice.add_trace(
    go.Scatter(
        x=[2.75, 5],
        y=[4.5, 0],
        mode="lines",
        line={"color": "#ff7f0e"},
        name="2x + y ≤ 10 (assembly hours)",
        hoverinfo="skip",
    )
)
grid = [(x, y) for x in range(7) for y in range(5)]
feasible_pts = [(x, y) for x, y in grid if x + 3 * y <= 12 and 2 * x + y <= 10]
infeasible_pts = [pt for pt in grid if pt not in feasible_pts]
lattice.add_trace(
    go.Scatter(
        x=[x for x, _ in feasible_pts],
        y=[y for _, y in feasible_pts],
        customdata=[3 * x + 5 * y for x, y in feasible_pts],
        mode="markers",
        marker={
            "color": "#2ca02c",
            "size": 9,
            "line": {"color": "white", "width": 1},
        },
        name="integer feasible points",
        hovertemplate="(%{x}, %{y})<br>profit %{customdata}<extra></extra>",
    )
)
lattice.add_trace(
    go.Scatter(
        x=[x for x, _ in infeasible_pts],
        y=[y for _, y in infeasible_pts],
        mode="markers",
        marker={"color": "rgba(128,128,128,0.45)", "size": 9},
        name="integer infeasible points",
        hovertemplate="(%{x}, %{y})<br>infeasible<extra></extra>",
    )
)
lattice.add_trace(
    go.Scatter(
        x=[3.6],
        y=[2.8],
        mode="markers",
        marker={"color": "#d62728", "size": 18, "symbol": "star"},
        name="LO relaxation optimum",
        hovertemplate="LO relaxation optimum (3.6, 2.8)<br>profit 24.8"
        "<extra></extra>",
    )
)
x_int, y_int = dec_vars["bookcase"].value(), dec_vars["desk"].value()
lattice.add_trace(
    go.Scatter(
        x=[x_int],
        y=[y_int],
        mode="markers",
        marker={"color": "#9467bd", "size": 15, "symbol": "diamond"},
        name="ILO optimum",
        hovertemplate="ILO optimum (%{x}, %{y})<br>profit 24<extra></extra>",
    )
)
lattice.update_layout(**PLOT_LAYOUT, height=560)
lattice.update_xaxes(range=[-0.2, 6.2], title="bookcases x", **AXIS_STYLE)
lattice.update_yaxes(range=[-0.2, 4.7], title="desks y", **AXIS_STYLE)
lattice.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #lattice-plot
# :label: fig-lattice
#
# Feasible region of the integer product-mix problem with its integer points, the optimum of the LO relaxation (star) and the ILO optimum (diamond). Hover over a point to see its coordinates and profit.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-9
#
# Solve the larger LO problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb#larger-lo-example) again, once
# requiring all variables integer and once requiring them binary. Compare the optimal
# objective values with the continuous one.
# :::

# %% [markdown]
# (branch-and-bound)=
# ## Branch and Bound
#
# A common method for solving integer linear optimization (ILO) problems is **branch and bound**. The basic idea is to start by just ignoring the integer restrictions. More precisely, we allow integer variables to be continuous while retaining all their other bounds and constraints. For example, instead of requiring a binary variable to satisfy $x_i \in \{0,1\}$, we allow it to take any value between zero and one:
# $$
# 0 \leq x_i \leq 1.
# $$
# Likewise, instead of requiring $x_i \in \{0,1,2,\ldots\}$, we only require:
# $$
# x_i \geq 0.
# $$
# This produces the **LO relaxation**: the same optimization problem, but without the requirement that certain variables must be integers.
#
# The optimal solution to the LO relaxation usually contains non-integer values that should be integers. Such a solution is therefore infeasible for the original ILO problem, but it is still useful in two ways. First, for a maximization problem, it provides an **upper bound (UB)**: the best possible integer solution cannot have a higher objective value than the relaxed solution. Second, it provides a starting point for further search by **branching** on non-integer variables one by one.
#
# Branching on an LO relaxation that yields an infeasible ILO solution works as follows. Suppose the relaxed solution contains $x_j = 2.5$, while $x_j$ should be an integer. We then branch the original root problem into two subproblems based on the LO relaxation of the original problem:
#
# - First subproblem: original problem but with extra constraint $x_j \leq 2$;
# - Second subproblem: original problem but with extra constraint $x_j \geq 3$;
#
# Together, these two subproblems cover all possible integer values of $x_j$, but neither allows $x_j = 2.5$ anymore. We then solve the LO relaxation of each subproblem (independently) in the same manner. If either relaxation again yields a non-integer variable that must be integer, we apply the same strategy and branch again, et cetera. To illustrate this, suppose that solving the relaxation for the subproblem with $x_j \leq 2$ leads to a solution with $x_k = 5.6$. We then branch this subproblem again to get one subproblem with constraints $x_j \leq 2$ and $x_k \leq 5$, and one subproblem with $x_j \leq 2$ and $x_k \geq 6$. Indeed, both subproblems still have the previous constraint $x_j \leq 2$.
#
# At first sight, this may seem like enumerating all possible integer values. The key difference is that branch and bound uses bounds to avoid exploring subproblems that cannot improve the best integer solution found so far. Hence the name branch and bound.
#
# Any feasible integer solution to the original ILO problem provides a **lower bound (LB)** for a maximization problem. Along the way, we keep track of the best original ILO solution, called the **incumbent**. Its objective value is the current best/largest LB. On the other hand, as mentioned before, the LO relaxation of a subproblem is always a UB for the best integer-feasible solution at the subproblem. So if the UB at a subproblem is smaller than the current best LB, we don't need to branch further on that subproblem, as it will never lead to a better ILO solution. This saves computation time.
#
# Let us formalize this method further. The branch and bound method keeps track of (1) a pool of (sub)problems whose LO relaxation is not solved yet and which may still lead to an optimal solution, and (2) the current best solution with the largest/best LB so far. Initially, this pool contains only the original problem, and the LB is set to $-\infty$. As long as the pool is not empty, choose a (sub)problem from the pool, remove it, and solve its LO relaxation. This solve leads to one of three outcomes:
#
# 1. The found solution is infeasible for the ILO problem, and its objective value is a UB:
#    1. If UB $\leq$ largest LB: We eliminate this subproblem (we will not find a better solution here; this can by default not happen for the original problem).
#    2. If UB $>$ largest LB: Pick a variable from the solution that is non-integer but should be. Branch the (sub)problem on this variable into two new (sub)problems and add them to the pool.
# 2. The found solution is feasible for the ILO problem and gives a new LB:
#    1. If LB $>$ largest LB: Update the new best LB and see whether we can eliminate subproblems that were not eliminated yet. In particular, for all non-eliminated subproblems for which we obtained a UB, check whether their UB $<$ largest LB; if so, eliminate the subproblem and remove all its descendants still in the pool.
#    2. If LB $\leq$ largest LB: Eliminate this subproblem (nothing to branch further here).
# 3. There is no feasible solution for the specific (sub)problem: Eliminate this subproblem as it cannot have an ILO feasible solution.
#
# Keep repeating this procedure until the pool is empty. Once the pool is empty, we are guaranteed that the solution with the current best LB is optimal.
#
# The process can be visualized as a tree. The original problem is the root. Each time we branch, we split one (sub)problem into two child subproblems. The pool contains the subproblems that have not yet been eliminated or solved.
#
# We did not discuss how to choose two things: which decision variable to branch on (since an LO relaxation solution typically has many non-integer variables that should be integer) and which (sub)problem from the pool to pick first. The choice affects performance, and what works best is a research topic on its own and outside the scope of this course. In this course, just make a choice; we always know that in the end we will find the optimal solution.
#
# Let us apply this idea to the integer product-mix problem to illustrate its workings. We number the steps in the order in which we take them:
#
# - **Step 1: root problem.** The LO relaxation has optimum $\left(3.6, 2.8\right)$ with objective value $24.8$. This is a UB, but the solution is not integer in both decision variables and thus infeasible. Just pick a decision variable to branch on. We branch on $y = 2.8$, which adds the subproblems with $y \leq 2$ and $y \geq 3$ to the pool.
# - **Step 2: left subproblem,** the root problem with $y \leq 2$. Its relaxation has optimum $\left(4, 2\right)$ with value $22$. This solution is integer, so it is feasible for the ILO problem and thus holds the current best LB of $22$. No need to branch further from this subproblem.
# - **Step 3: right subproblem,** the root problem with $y \geq 3$. Its relaxation has optimum $\left(3, 3\right)$ with value $24$. This is also feasible for the ILO problem, so it improves the best LB to $24$, and we can replace the previous best solution with the solution $\left(3, 3\right)$. No need to branch further from this subproblem.
# - **Step 4: left subproblem is suboptimal.** Its value of $22$ is below the new best LB of $24$, so the solution $\left(4, 2\right)$ is no longer the best one.
# - **Step 5: optimum.** The pool is now empty (no promising (sub)problems left to explore), so we are done, and the current best solution of $\left(3,3\right)$ with profit $24$ must be the optimal ILO solution.
#
# The figure below shows these steps as a branch-and-bound tree. Each node is a (sub)problem: its header gives the step in which its LO relaxation is solved, the objective value, and whether that value is a UB or a LB, and the line below gives the relaxation's optimal solution $(x, y)$. The labels on the edges are the constraints added by branching.

# %% tags=["hide-input"]
# header and body colors per node type, and colors of the status boxes
NODE_COLORS = {
    "UB": ("#5b9bd5", "#d6dce5"),
    "LB": ("#70ad47", "#d9e7cf"),
    "infeasible": ("#e00000", "#fbd5b5"),
}
STATUS_COLORS = {"pruned": "#e00000", "optimum": "#375623"}
EDGE_COLOR = "#5b9bd5"


def draw_bb_tree(
    nodes, edges, status, xlim, ylim, figsize, box_width=3.6, font_size=11
):
    """Draw a branch-and-bound tree.

    nodes: name -> (x, y, header, solution, node type)
    edges: (parent, child, branching constraint) tuples
    status: name -> (text, status type) for a box below the node

    The background is transparent and every text sits on its own box, so
    the figure reads the same in the light and the dark site theme.
    """
    box_height, gap = 0.55, 0.08
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for parent, child, branch_label in edges:
        x_from, y_from = nodes[parent][:2]
        x_to, y_to = nodes[child][:2]
        y_from -= 1.5 * box_height + gap
        y_to += 0.5 * box_height
        ax.plot([x_from, x_to], [y_from, y_to], color=EDGE_COLOR, lw=1.2)
        ax.text(
            (x_from + x_to) / 2,
            (y_from + y_to) / 2,
            branch_label,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": EDGE_COLOR,
            },
        )
    for node, (x_c, y_c, header, body, node_type) in nodes.items():
        dark, light = NODE_COLORS[node_type]
        boxes = [(header, dark, "white", "bold"), (body, light, "black", "")]
        if node in status:
            text, status_type = status[node]
            boxes.append((text, STATUS_COLORS[status_type], "white", "bold"))
        for row, (text, face, text_color, font_weight) in enumerate(boxes):
            y_row = y_c - row * (box_height + gap)
            ax.add_patch(
                Rectangle(
                    (x_c - box_width / 2, y_row - box_height / 2),
                    box_width,
                    box_height,
                    facecolor=face,
                    edgecolor="none",
                )
            )
            ax.text(
                x_c,
                y_row,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontweight=font_weight or "normal",
                fontsize=font_size,
            )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    plt.show()


# %% tags=["remove-cell"] label="bb-product-mix"
draw_bb_tree(
    nodes={
        "root": (
            5.0,
            5.0,
            "Step 1: 24.8 (UB)",
            "3.6, 2.8",
            "UB",
        ),
        "left": (
            2.2,
            2.4,
            "Step 2: 22 (LB)",
            "4, 2",
            "LB",
        ),
        "right": (
            7.8,
            2.4,
            "Step 3: 24 (LB)",
            "3, 3",
            "LB",
        ),
    },
    edges=[
        ("root", "left", r"$y \leq 2$"),
        ("root", "right", r"$y \geq 3$"),
    ],
    status={
        "left": ("Step 4: 22 < 24 (subopt)", "pruned"),
        "right": ("Step 5: optimum", "optimum"),
    },
    xlim=(0, 10),
    ylim=(0.6, 5.4),
    figsize=(8, 3.9),
)

# %% [markdown]
# :::{figure} #bb-product-mix
# :label: fig-bb-product-mix
#
# Branch-and-bound tree for the integer product-mix problem (UB = upper bound from the LO relaxation, LB = lower bound from a feasible integer solution, subopt = suboptimal solution).
# :::

# %% [markdown]
# In this example, we needed only three LO relaxations, rather than checking every possible integer combination. Modern integer-optimization solvers use this basic branch-and-bound idea, often enhanced with additional techniques. More on this later.
#
# :::{exercise}
# :label: ex-bb-branch-x
#
# Solve the integer product-mix problem with branch and bound again, but now branch on $x$ instead of $y$ in step 1. Solve each LO relaxation graphically, number the steps, and draw the branch-and-bound tree in the same style as the figure above. Do you find the same optimal solution, and how many LO relaxations do you need this time?
# :::

# %% [markdown]
# ## The Knapsack Problem
#
# The archetypal binary ILO problem is the **knapsack problem**: from a set of items, each with
# a *reward* and a *weight*, choose a subset of maximum total reward whose total weight fits
# a given capacity. Applications include which items to load in a truck, cutting stock in a steel
# plant, and simple forms of portfolio selection.
#
# The knapsack problem is interesting in its own right, and its simple structure also makes it a good problem to see branch and bound at work. We first solve an instance with pulp and then solve the same instance by hand with branch and bound.
#
# Let us consider a concrete example with $n=4$ numbered items:
#
# | item $i$     | 1  | 2 | 3  | 4 |
# |--------------|----|---|----|---|
# | reward $r_i$ | 15 | 9 | 10 | 5 |
# | weight $w_i$ | 1  | 3 | 5  | 4 |
#
# In here, the reward and weight of the $i$th item is denoted by $r_i$ and $w_i$, respectively. The capacity $C = 8$. Furthermore, define the binary decision variable $x_i$ as 1 if we take item $i$ and 0 otherwise. The ILO problem can then be stated as follows:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{i=1}^n r_i x_i \\
# \text{subject to} \quad & \sum_{i=1}^n w_i x_i \le C \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# This problem can be solved in pulp as follows.

# %%
# data
reward = [15, 9, 10, 5]
weight = [1, 3, 5, 4]
capacity = 8
n_items = len(reward)

# modeling
knapsack = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
take = [
    pulp.LpVariable(name=f"x_{i + 1}", cat="Binary") for i in range(n_items)
]
knapsack += pulp.lpSum(reward[i] * take[i] for i in range(n_items))
knapsack += pulp.lpSum(weight[i] * take[i] for i in range(n_items)) <= capacity

# solve and print solution
knapsack.solve(pulp.PULP_CBC_CMD(msg=False))
print("take items:", [i + 1 for i in range(n_items) if take[i].value() == 1])
print("total reward:", knapsack.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-10
#
# Take the knapsack solution pulp found above. Verify by hand that no single swap (adding
# one currently-excluded item and removing whatever is needed to stay within capacity)
# improves the total reward.
# :::

# %% [markdown]
# (knapsack-lo-relaxation)=
# ### Solving the LO Relaxation
#
# Now let us solve our example with the branch and bound method. Every step of branch and bound solves an LO relaxation, so we first need to know how to do that. For the knapsack problem this turns out to be easy, and no LO solver is needed.
#
# In the LO relaxation, $x_i \in \{0, 1\}$ becomes $0 \leq x_i \leq 1$: we may take any fraction of an item, and a fraction $x_i$ of item $i$ gives reward $r_i x_i$ and uses capacity $w_i x_i$. Each unit of capacity should then go to the item that gives the most reward per unit of weight. So we sort the items in decreasing order of their **reward-to-weight ratio** $r_i / w_i$ and fill the knapsack in that order, taking each item completely as long as it fits. The first item that no longer fits completely is taken for the fraction that fills the remaining capacity, and all later items are left out. As a result, at most one variable in the optimal solution of the LO relaxation is fractional.
#
# For our example, the ratios are $15/1 = 15$, $9/3 = 3$, $10/5 = 2$ and $5/4 = 1.25$, so the items are already in decreasing order of ratio. Filling the knapsack in that order goes as follows:
#
# | item $i$ | ratio $r_i / w_i$ | weight $w_i$ | fraction taken $x_i$ | capacity left |
# |---|---|---|---|---|
# | 1 | 15 | 1 | 1 | 7 |
# | 2 | 3 | 3 | 1 | 4 |
# | 3 | 2 | 5 | 4/5 | 0 |
# | 4 | 1.25 | 4 | 0 | 0 |
#
# Items 1 and 2 fit completely. Item 3 does not: only 4 of its weight of 5 fits, so we take a fraction $4/5$ of it, and there is no capacity left for item 4. The optimal solution of the LO relaxation is therefore $(1, 1, 0.8, 0)$ with objective value $15 + 9 + 0.8 \cdot 10 = 32$.
#
# In branch and bound, some variables are fixed to 0 or 1 in a subproblem. The same rule then still applies: items fixed to 1 are packed first, items fixed to 0 are left out, and the remaining items fill the remaining capacity in decreasing order of ratio. If the items fixed to 1 already exceed the capacity, the subproblem is infeasible.

# %% [markdown]
# ### Branch and Bound for the Knapsack Problem
#
# With the LO relaxation in hand, we solve our example with branch and bound. Since the LO relaxation has at most one fractional variable, there is only one variable to branch on in each step, and branching on a binary variable means fixing it to 0 in one subproblem and to 1 in the other. We use the same step numbering as before:
#
# - **Step 1: root problem.** The LO relaxation has optimum $(1, 1, 0.8, 0)$ with value $32$, a UB. We branch on $x_3 = 0.8$.
# - **Step 2: subproblem $x_3 = 0$.** Without item 3, items 1, 2 and 4 fit exactly: the relaxation has optimum $(1, 1, 0, 1)$ with value $29$. This solution is integer, so it is feasible for the ILO problem and gives the current best LB of $29$. No need to branch further from this subproblem.
# - **Step 3: subproblem $x_3 = 1$.** Item 3 uses 5 of the capacity, item 1 then fits, and $2/3$ of item 2 fills the rest: optimum $(1, 2/3, 1, 0)$ with value $31$. This UB is larger than the best LB of $29$, so this subproblem may still contain a better solution. We branch on $x_2$.
# - **Step 4: subproblem $x_3 = 1$, $x_2 = 0$.** Items 3 and 1 fit, and half of item 4 fills the rest: optimum $(1, 0, 1, 0.5)$ with value $27.5$, a UB.
# - **Step 5: eliminate.** The UB of $27.5$ is below the best LB of $29$, so this subproblem cannot contain a better solution and we eliminate it without branching.
# - **Step 6: subproblem $x_3 = 1$, $x_2 = 1$.** Items 3 and 2 use the full capacity, so item 1 no longer fits: optimum $(0, 1, 1, 0)$ with value $19$. This solution is integer and thus a LB.
# - **Step 7: suboptimal.** The LB of $19$ is below the best LB of $29$.
# - **Step 8: optimum.** The pool is now empty, so the best solution found, $(1, 1, 0, 1)$ with reward $29$ from step 2, is optimal.
#
# The figure below shows the branch-and-bound tree.

# %% tags=["remove-cell"] label="bb-knapsack"
draw_bb_tree(
    nodes={
        "root": (
            7.2,
            5.0,
            "Step 1: 32 (UB)",
            "1, 1, 0.8, 0",
            "UB",
        ),
        "x3=0": (
            2.5,
            2.6,
            "Step 2: 29 (LB)",
            "1, 1, 0, 1",
            "LB",
        ),
        "x3=1": (
            10.2,
            2.6,
            "Step 3: 31 (UB)",
            "1, 2/3, 1, 0",
            "UB",
        ),
        "x2=0": (
            7.9,
            0.2,
            "Step 4: 27.5 (UB)",
            "1, 0, 1, 0.5",
            "UB",
        ),
        "x2=1": (
            12.5,
            0.2,
            "Step 6: 19 (LB)",
            "0, 1, 1, 0",
            "LB",
        ),
    },
    edges=[
        ("root", "x3=0", r"$x_3 = 0$"),
        ("root", "x3=1", r"$x_3 = 1$"),
        ("x3=1", "x2=0", r"$x_2 = 0$"),
        ("x3=1", "x2=1", r"$x_2 = 1$"),
    ],
    status={
        "x3=0": ("Step 8: optimum", "optimum"),
        "x2=0": ("Step 5: 27.5 (UB) < 29 (LB)", "pruned"),
        "x2=1": ("Step 7: 19 < 29 (subopt)", "pruned"),
    },
    xlim=(0, 15),
    ylim=(-1.45, 5.4),
    figsize=(11, 5),
    box_width=4.5,
)

# %% [markdown]
# :::{figure} #bb-knapsack
# :label: fig-bb-knapsack
#
# Branch-and-bound tree for the knapsack example (UB = upper bound from the LO relaxation, LB = lower bound from a feasible integer solution, subopt = suboptimal solution).
# :::

# %% [markdown]
# (general-formulation)=
# ## General Formulation
#
# Having now seen both LO and ILO in action, we can step back and write down the general
# form both fit into. An LO problem with $n$ decision variables and $m$ constraints is
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

# %% [markdown]
# (why-linearity-matters)=
# ## Why Linearity Matters, and What Integrality Buys Back
#
# Linearity is what makes LO efficiently solvable: written in the
# [general form above](#general-formulation), the feasible region is a convex polyhedron,
# an optimum sits at a corner, and a local optimum is automatically global. As soon as the
# objective or a constraint is nonlinear, both of those break:
#
# - **Nonlinear objective.** Maximize $x_1 x_2$ subject to $x_1 + x_2 \le 1$,
#   $x_1, x_2 \ge 0$. The optimum is $(0.5, 0.5)$, in the *interior* of an edge, not at a
#   corner.
# - **Nonlinear constraint.** Maximize $x_1 + x_2$ subject to $\min(x_1, x_2) = 0$,
#   $x_1 \le 2$, $x_2 \le 1$. The feasible set is two line segments meeting at the origin
#   (either $x_1 = 0$ or $x_2 = 0$). It has two *local* optima, $(2, 0)$ and $(0, 1)$; you
#   cannot be sure which is global without checking both. More generally, the simplex
#   reasoning from [Linear Optimization](lecture8_linear-optimization.ipynb) breaks down
#   whenever the feasible region is not a convex polyhedron: you are not sure you have
#   found the best solution until you have checked every local optimum, which is usually
#   intractable.
#
# Nonlinear optimization therefore needs slower, less reliable algorithms. But there is a
# large and useful middle ground: integer constraints. On the one hand, requiring
# $x_i \in \{0, 1, 2, \dots\}$ is itself a nonlinear constraint, and it does make a problem
# harder to solve, as branch and bound's extra work above shows. On the other hand, unlike
# general nonlinearities, it is not *intractably* so, and many other nonlinearities (an
# either/or choice, a fixed cost that applies only when an activity is used, a "this
# constraint holds only if..." condition) can be expressed with integer (usually binary)
# variables and otherwise-linear constraints, and then solved with branch and bound. That is
# why so much modeling effort goes into casting a problem as ILO, and why it is worth
# treating as its own class rather than lumping it in with general nonlinear optimization.
# [Set Covering and Shift Scheduling](lecture9_covering.ipynb) and
# [Machine Scheduling](lecture9_machine-scheduling.ipynb) show many of those tricks in
# practice.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.4 "Integer Problems."
