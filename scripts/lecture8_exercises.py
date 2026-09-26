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
# description: "Homework exercises and further exercises for Lecture 8 on linear and integer optimization."
# thumbnail: null
# ---
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
import plotly.graph_objects as go
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

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
    "margin": {"l": 60, "r": 20, "t": 30, "b": 50},
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
LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def region(corners, objective, name="corner point"):
    """Shaded feasible region; hovering a corner shows its objective value."""
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    fill = go.Scatter(
        x=xs,
        y=ys,
        fill="toself",
        fillcolor="rgba(31,119,180,0.15)",
        mode="none",
        name="feasible region",
        hoverinfo="skip",
    )
    points = go.Scatter(
        x=xs,
        y=ys,
        customdata=[objective(x, y) for x, y in corners],
        mode="markers",
        marker={"color": "rgba(31,119,180,0.6)", "size": 8},
        name=name,
        hovertemplate="(%{x:.3~f}, %{y:.3~f})<br>objective %{customdata:.3~f}"
        "<extra></extra>",
    )
    return [fill, points]


def line(a, b, c, xlim, ylim, name, color, dash=None):
    """The line a x + b y = c within the plotted area."""
    if b == 0:
        xs, ys = [c / a, c / a], list(ylim)
    else:
        xs = list(xlim)
        ys = [(c - a * x) / b for x in xs]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "dash": dash},
        name=name,
        hoverinfo="skip",
    )


def point(x, y, name, hover, color="#d62728", symbol="star", size=18):
    """A highlighted point with its own hover text."""
    return go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker={"color": color, "size": size, "symbol": symbol},
        name=name,
        hovertemplate=hover + "<extra></extra>",
    )


def show_region(fig, xlim, ylim, xtitle="x", ytitle="y", height=520):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(range=list(xlim), title=xtitle, **AXIS_STYLE)
    fig.update_yaxes(range=list(ylim), title=ytitle, **AXIS_STYLE)
    fig.show(config=PLOT_CONFIG)


# %% tags=["remove-cell"] label="hw-8-1b"
lim_x, lim_y = (-0.2, 8.0), (-0.2, 5.5)
fig = go.Figure(
    region([(0, 0), (5, 0), (5, 2.5), (3, 3.5), (0, 3.5)], lambda x, y: x + y)
)
fig.add_trace(line(1, 0, 5, lim_x, lim_y, "x ≤ 5", LINE_COLORS[0]))
fig.add_trace(line(0, 1, 3.5, lim_x, lim_y, "y ≤ 3.5", LINE_COLORS[1]))
fig.add_trace(line(1, 2, 10, lim_x, lim_y, "x + 2y ≤ 10", LINE_COLORS[2]))
fig.add_trace(line(1, 1, 7.5, lim_x, lim_y, "x + y = 7.5", GREY, "dash"))
fig.add_trace(point(5, 2.5, "optimum", "optimum (5, 2.5)<br>objective 7.5"))
show_region(fig, lim_x, lim_y)

# %% tags=["remove-cell"] label="hw-8-1c"
fig = go.Figure(
    region([(0, 0), (5, 0), (5, 2.5), (3, 3.5), (0, 3.5)], lambda x, y: x + y)
)
for level in range(4):
    x_max = min(5, 10 - 2 * level)
    fig.add_trace(
        go.Scatter(
            x=[0, x_max],
            y=[level, level],
            mode="lines",
            line={"color": LINE_COLORS[2], "width": 5},
            name="feasible (y integer)",
            legendgroup="segments",
            showlegend=level == 0,
            hovertemplate=f"y = {level}, 0 ≤ x ≤ {x_max}<br>"
            f"best objective {x_max + level}<extra></extra>",
        )
    )
fig.add_trace(line(1, 1, 7, lim_x, lim_y, "x + y = 7", GREY, "dash"))
fig.add_trace(
    point(5, 2.5, "LO relaxation optimum", "LO relaxation optimum (5, 2.5)")
)
fig.add_trace(
    go.Scatter(
        x=[5, 4],
        y=[2, 3],
        mode="markers",
        marker={"color": "#9467bd", "size": 15, "symbol": "diamond"},
        name="optimal solutions",
        hovertemplate="optimal (%{x}, %{y})<br>objective 7<extra></extra>",
    )
)
show_region(fig, lim_x, lim_y)

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
# Solution of [](#hw-8-1), part b: the feasible region of part a, with the objective line $x + y = 7.5$ through the optimum
# $(5, 2.5)$.
# Hover over a corner point to see its objective value.
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
# Solution of [](#hw-8-1), part c: with $y$ integer and $x$ continuous, the feasible solutions lie on the
# horizontal segments. The objective line $x + y = 7$ touches them in $(5, 2)$ and
# $(4, 3)$.
# Hover over a segment or point for its objective value.
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
lim_x, lim_y = (-0.2, 5.5), (-0.2, 8.0)
fig = go.Figure(
    region([(0, 0), (3.75, 0), (2.5, 2.5), (0, 5)], lambda x, y: 1.5 * x + y)
)
fig.add_trace(line(1, 1, 5, lim_x, lim_y, "x + y ≤ 5 (flour)", LINE_COLORS[0]))
fig.add_trace(
    line(2, 1, 7.5, lim_x, lim_y, "2x + y ≤ 7.5 (yeast)", LINE_COLORS[1])
)
fig.add_trace(
    line(1.5, 1, 6.25, lim_x, lim_y, "1.5x + y = 6.25", GREY, "dash")
)
fig.add_trace(
    point(2.5, 2.5, "optimum", "optimum (2.5, 2.5)<br>objective 6.25")
)
show_region(fig, lim_x, lim_y)

# %% tags=["remove-cell"] label="hw-8-2d"
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("case z = 0", "case z > 0, so x = 2"),
    horizontal_spacing=0.12,
)
left = region([(0, 0), (2, 0), (2, 3), (0, 5)], lambda x, y: 1.5 * x + y)
left += [
    line(1, 1, 5, lim_x, lim_y, "x + y ≤ 5", LINE_COLORS[0]),
    line(2, 1, 7.5, lim_x, lim_y, "2x + y ≤ 7.5", LINE_COLORS[1]),
    line(1, 0, 2, lim_x, lim_y, "x ≤ 2", LINE_COLORS[2]),
    line(1.5, 1, 6, lim_x, lim_y, "1.5x + y = 6", GREY, "dash"),
    point(2, 3, "optimum", "optimum (x, y) = (2, 3)<br>profit 6"),
]
lim_y_case, lim_z_case = (-0.1, 4.0), (-0.1, 3.0)
right = region(
    [(0, 0), (3, 0), (2.5, 0.5), (0, 1.75)],
    lambda y, z: 3 + y + 0.5 * z,
    name="corner point (y, z)",
)
right += [
    line(1, 1, 3, lim_y_case, lim_z_case, "y + z ≤ 3", LINE_COLORS[0]),
    line(1, 2, 3.5, lim_y_case, lim_z_case, "y + 2z ≤ 3.5", LINE_COLORS[1]),
    line(1, 0.5, 3, lim_y_case, lim_z_case, "3 + y + 0.5z = 6", GREY, "dash"),
    point(3, 0, "optimum", "optimum (y, z) = (3, 0)<br>profit 6"),
]
for trace in left:
    fig.add_trace(trace, row=1, col=1)
for trace in right:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(**PLOT_LAYOUT, height=540)
fig.update_layout(margin={"t": 60})
fig.update_xaxes(range=list(lim_x), title="x", row=1, col=1, **AXIS_STYLE)
fig.update_yaxes(range=list(lim_y), title="y", row=1, col=1, **AXIS_STYLE)
fig.update_xaxes(range=list(lim_y_case), title="y", row=1, col=2, **AXIS_STYLE)
fig.update_yaxes(range=list(lim_z_case), title="z", row=1, col=2, **AXIS_STYLE)
fig.update_annotations(font={"color": TEXT_COLOR})
fig.show(config=PLOT_CONFIG)

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
# Solution of [](#hw-8-2), part b: the feasible region of part a, with the objective line $1.5x + y = 6.25$ through the
# optimum $(2.5, 2.5)$.
# Hover over a corner point to see its objective value.
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
#    Solution of [](#hw-8-2), part d: the two cases. Left: $z = 0$, the region of part b with $x \le 2$ added.
#    Right: $z > 0$, so $x = 2$, and the remaining problem in $y$ and $z$.
#    Hover over a corner point to see its objective value.
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

# %% tags=["remove-cell"] label="hw-8-3c"
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


draw_bb_tree(
    nodes={
        "s1": (
            6.1,
            7.5,
            "Step 1: 14.75 (UB)",
            "1, 0, 1, 0.75",
            "UB",
        ),
        "s2": (
            2.4,
            5.0,
            "Step 2: 12 (LB)",
            "1, 1, 1, 0",
            "LB",
        ),
        "s3": (
            9.8,
            5.0,
            "Step 3: 14.5 (UB)",
            "0.5, 0, 1, 1",
            "UB",
        ),
        "s4": (
            4.8,
            2.5,
            "Step 4: 13.5 (UB)",
            "0, 0.5, 1, 1",
            "UB",
        ),
        "s9": (
            14.4,
            2.5,
            "Step 9: 14 (UB)",
            "1, 0, 0.75, 1",
            "UB",
        ),
        "s5": (
            2.4,
            0.0,
            "Step 5: 13 (LB)",
            "0, 0, 1, 1",
            "LB",
        ),
        "s7": (
            7.2,
            0.0,
            "Step 7: 12 (UB)",
            "0, 1, 0.75, 1",
            "UB",
        ),
        "s10": (
            12.0,
            0.0,
            "Step 10: 9 (LB)",
            "1, 1, 0, 1",
            "LB",
        ),
        "s12": (
            16.8,
            0.0,
            "Step 12: infeasible",
            r"$x_1 = x_3 = x_4 = 1$",
            "infeasible",
        ),
    },
    edges=[
        ("s1", "s2", r"$x_4 = 0$"),
        ("s1", "s3", r"$x_4 = 1$"),
        ("s3", "s4", r"$x_1 = 0$"),
        ("s3", "s9", r"$x_1 = 1$"),
        ("s4", "s5", r"$x_2 = 0$"),
        ("s4", "s7", r"$x_2 = 1$"),
        ("s9", "s10", r"$x_3 = 0$"),
        ("s9", "s12", r"$x_3 = 1$"),
    ],
    status={
        "s2": ("Step 6: 12 < 13 (subopt)", "pruned"),
        "s5": ("Step 13: optimum", "optimum"),
        "s7": ("Step 8: 12 (UB) < 13 (LB)", "pruned"),
        "s10": ("Step 11: 9 < 13 (subopt)", "pruned"),
    },
    xlim=(0, 19.2),
    ylim=(-1.75, 7.9),
    figsize=(12, 5.8),
    box_width=4.4,
    font_size=10,
)

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
# :::{figure} #hw-8-3c
# :label: fig-hw-8-3c
#
# Solution of [](#hw-8-3), part c: branch-and-bound tree (UB = upper bound from the LO
# relaxation, LB = lower bound from a feasible integer solution, subopt = suboptimal
# solution). The infeasible subproblem of step 12 has no solution; its second line shows
# the variables fixed to 1.
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
lim_x, lim_y = (-0.2, 4.5), (-0.2, 5.0)
fig = go.Figure(
    region([(0, 0), (3, 0), (16 / 7, 20 / 7), (0, 4)], lambda x, y: x + y)
)
fig.add_trace(line(4, 1, 12, lim_x, lim_y, "4x + y ≤ 12", LINE_COLORS[0]))
fig.add_trace(line(2, 4, 16, lim_x, lim_y, "2x + 4y ≤ 16", LINE_COLORS[1]))
fig.add_trace(line(1, 1, 36 / 7, lim_x, lim_y, "x + y = 36/7", GREY, "dash"))
int_points = [
    (i, j)
    for i in range(5)
    for j in range(6)
    if 4 * i + j <= 12 and 2 * i + 4 * j <= 16
]
fig.add_trace(
    go.Scatter(
        x=[i for i, _ in int_points],
        y=[j for _, j in int_points],
        customdata=[i + j for i, j in int_points],
        mode="markers",
        marker={"color": LINE_COLORS[2], "size": 9},
        name="integer feasible points",
        hovertemplate="(%{x}, %{y})<br>objective %{customdata}<extra></extra>",
    )
)
fig.add_trace(
    point(
        16 / 7,
        20 / 7,
        "LO optimum",
        "LO optimum (16/7, 20/7)<br>objective 36/7 ≈ 5.14",
    )
)
fig.add_trace(
    point(
        2,
        3,
        "integer optimum",
        "integer optimum (2, 3)<br>objective 5",
        color="#9467bd",
        symbol="diamond",
        size=15,
    )
)
show_region(fig, lim_x, lim_y)

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
# Solution of [](#hw-8-4), part b: the feasible region of part a, with the objective line through the LO optimum
# $(16/7, 20/7)$ and the feasible integer points used in part c.
# Hover over a point to see its objective value.
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
