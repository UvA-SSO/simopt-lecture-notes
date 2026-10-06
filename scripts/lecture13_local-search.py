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
# description: "Local search for simulation optimization over a large discrete set of solutions, illustrated on a newsvendor problem with a local optimum."
# thumbnail: null
# ---
# # Lecture 13: Local Search
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_local-search.ipynb)

# %% [markdown]
# [Ranking and Selection](lecture13_ranking-and-selection.ipynb) simulated every solution a number of times. When $S$ is too large for that, the third of the [four types of problems](lecture13_about-simopt.ipynb#four-types), we search through $S$ step by step instead, as in the [local search heuristic](lecture11_complexity-heuristics.ipynb#local-search-heuristic) of Lecture 11. This notebook gives a simple local search algorithm for simulation optimization and applies it to a newsvendor problem that has a local optimum.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - describe the local search algorithm for simulation optimization, and read and adapt its Python code
# - explain how noise lets stochastic local search escape a local optimum, unlike deterministic local search
# - explain why the algorithm returns the most visited solution

# %% [markdown]
# (local-search)=
# ## The Algorithm
#
# If $S$ is very large or even infinite (e.g., $\{1, 2, \dots\}$), we cannot start by simulating every $\pi \in S$ a number of times. There is often some structure that we can use, though: solutions that are close to each other have similar performance. As in [deterministic local search](lecture11_complexity-heuristics.ipynb#local-search-heuristic), we define a neighborhood $N(\pi) \subseteq S$ for every $\pi \in S$, for example the order sizes $\pi - 1$ and $\pi + 1$, or the points next to $\pi$ on a grid.
#
# The algorithm keeps track of a current solution $\pi^*$ and, for every solution $\pi$:
#
# - $n(\pi)$: the number of times $\pi$ has been simulated so far;
# - $y(\pi)$: the average of these $n(\pi)$ simulated outcomes.
#
# In every iteration it compares $\pi^*$ with a random neighbor, using one new run of each:
#
# - **Start:** set $n(\pi) = 0$ and $y(\pi) = 0$ for all $\pi \in S$, and choose an initial solution $\pi^*$.
# - **While** the simulation budget is not used up:
#   - choose $\pi'$ from $N(\pi^*)$ at random;
#   - simulate $\pi^*$ and $\pi'$ once each, and update $n$ and $y$ of both;
#   - if $y(\pi') > y(\pi^*)$, move to $\pi'$: set $\pi^* = \pi'$.
# - **Return:** the most visited solution, $\arg\max_{\pi} n(\pi)$.
#
# This is one of the simplest algorithms there is; the scientific literature describes many more advanced ones. In practice, the result usually improves if the algorithm is run several times from different initial solutions.
#
# Compare it with deterministic local search, where we can compute the objective $g(\pi)$ exactly: that algorithm moves to $\pi'$ only if $g(\pi') > g(\pi^*)$, and stops when no neighbor is better. It always ends in a local optimum, and it stays there. In stochastic local search, the comparison uses noisy averages, and a neighbor that is in fact worse can look better by chance, especially when it has only been simulated a few times. This randomness leads to more exploration of $S$: it can take the search out of a local optimum where deterministic local search gets stuck forever. The price is that the search sometimes wanders to a worse solution. That is why the algorithm returns the most visited solution and not the one with the highest average $y(\pi)$: a solution that was visited often had many chances to be beaten by a neighbor and survived them, while a solution with a high average based on a few lucky runs is not reliable.

# %% [markdown]
# (local-search-example)=
# ## Example: A Newsvendor with a Volume Discount
#
# A newsvendor sells papers for €1 each. Demand $X$ is Poisson distributed with mean 15, and there is room for 20 papers, so $S = \{0, 1, \dots, 20\}$. A paper costs €0.60, but the supplier gives a volume discount: if the newsvendor buys more than 15 papers, every paper costs €0.50. The neighborhood is $N(\pi) = \{\pi - 1, \pi + 1\}$ for $0 < \pi < 20$, $N(0) = \{1\}$ and $N(20) = \{19\}$.
#
# Before we search, we look at the expected profit, which we can compute exactly for this small example (as in [Ranking and Selection](lecture13_ranking-and-selection.ipynb#newsvendor)). We only use it to check the search, which itself only sees simulated profits.

# %%
import numpy as np
from scipy import stats

price, mean_demand, max_order = 1, 15, 20


def unit_cost(order):
    return 0.6 if order <= 15 else 0.5


def simulate(order, rng):
    """Profit of one simulated day with the given order size."""
    return (
        price * min(rng.poisson(mean_demand), order) - unit_cost(order) * order
    )


orders = np.arange(max_order + 1)
true_value = np.array(
    [
        price * stats.poisson.sf(np.arange(o), mean_demand).sum()
        - unit_cost(o) * o
        for o in orders
    ]
)
print("expected profit:", true_value.round(2))

# %% label="ls-profit" tags=["remove-cell"]
import plotly.graph_objects as go

TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
    "height": 340,
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.22,
    },
}
AXIS_STYLE = {"gridcolor": "rgba(128,128,128,0.25)"}
# static pictures: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}

profit_fig = go.Figure()
profit_fig.add_trace(
    go.Scatter(
        x=orders,
        y=true_value,
        mode="markers",
        marker={"color": "#1f77b4", "size": 8},
        name="expected profit",
    )
)
for order, symbol, color, name in [
    (14, "circle-open", "#ff7f0e", "local optimum"),
    (16, "star", "#d62728", "global optimum"),
]:
    profit_fig.add_trace(
        go.Scatter(
            x=[order],
            y=[true_value[order]],
            mode="markers",
            marker={
                "symbol": symbol,
                "color": color,
                "size": 16,
                "line": {"width": 2, "color": color},
            },
            name=name,
        )
    )
profit_fig.update_layout(**PLOT_LAYOUT)
profit_fig.update_xaxes(title="order size π", dtick=2, **AXIS_STYLE)
profit_fig.update_yaxes(title="expected profit (€)", **AXIS_STYLE)
profit_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #ls-profit
# :label: fig-ls-profit
#
# Expected profit of every order size for the newsvendor with a volume discount above 15 papers.
# :::
#
# [](#fig-ls-profit) shows that order size 14 is a local optimum: both neighbors, 13 and 15, are worse. The discount makes order size 16 the global optimum. Deterministic local search that starts at $\pi^* = 0$ climbs to 14 and stops there, because 15 is worse than 14.
#
# The function below follows the algorithm line by line. The dictionaries `visits` and `average` hold $n(\pi)$ and $y(\pi)$; the average is updated with the new outcome without storing all earlier outcomes, using $y_{\text{new}} = y_{\text{old}} + (\text{outcome} - y_{\text{old}})/n$. Besides the most visited solution, the function returns the current solution after every iteration, so that we can plot the path of the search. As for the algorithms of Lecture 11, you should be able to read and follow this code; you do not need to be able to write it from scratch.


# %%
def neighbors(order):
    if order == 0:
        return [1]
    if order == max_order:
        return [max_order - 1]
    return [order - 1, order + 1]


def local_search(budget, start, rng):
    visits = {int(o): 0 for o in orders}
    average = {int(o): 0.0 for o in orders}
    current = start
    path = [current]
    # every iteration uses two runs: one of current, one of the neighbor
    for _ in range(budget // 2):
        candidate = int(rng.choice(neighbors(current)))
        for order in [current, candidate]:
            outcome = simulate(order, rng)
            visits[order] += 1
            average[order] += (outcome - average[order]) / visits[order]
        if average[candidate] > average[current]:
            current = candidate
        path.append(current)
    most_visited = max(visits, key=visits.get)
    return most_visited, path


# %% [markdown]
# We run the search three times from $\pi^* = 0$, with a budget of 2000 runs (1000 iterations) each and a different seed per run:

# %%
paths = {}
for seed in [8, 2, 6]:
    rng = np.random.default_rng(seed)
    result, path = local_search(2000, 0, rng)
    paths[seed] = path
    print(f"seed {seed}: most visited order size {result}")

# %% label="ls-paths" tags=["remove-cell"]
path_fig = go.Figure()
for seed, color in zip(paths, ["#1f77b4", "#ff7f0e", "#17becf"]):
    path_fig.add_trace(
        go.Scatter(
            x=np.arange(len(paths[seed])),
            y=paths[seed],
            mode="lines",
            line={"color": color, "width": 2, "shape": "hv"},
            name=f"seed {seed}",
        )
    )
path_fig.update_layout(**PLOT_LAYOUT)
path_fig.update_xaxes(title="iteration", **AXIS_STYLE)
path_fig.update_yaxes(
    title="current solution π*", range=[0, 20.5], dtick=2, **AXIS_STYLE
)
path_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #ls-paths
# :label: fig-ls-paths
#
# Current solution $\pi^*$ during three runs of local search, each with a budget of 2000 simulation runs.
# :::
#
# [](#fig-ls-paths) shows three different paths. All three climb quickly from 0 to the region around the local optimum 14. The run with seed 8 passes 15 by chance and reaches 16 within 50 iterations; later, a few lucky runs of larger order sizes take it up to 20 for a short while, but it returns to 16. The run with seed 6 stays around 13 and 14 for about 350 iterations before a lucky run of 15 takes it over the dip, after which it ends at 16 as well. The run with seed 2 never gets past the dip: it moves between 12 and 13, whose expected profits are close, and 14 and 15 rarely look better.
#
# A single run of the search can thus end in the local optimum. To see how often that happens, we repeat the search 300 times:

# %%
rng = np.random.default_rng(1)
results = [local_search(2000, 0, rng)[0] for _ in range(300)]
found, counts = np.unique(results, return_counts=True)
for order, count in zip(found, counts):
    print(f"order size {order}: {count} of the {len(results)} runs")

# %% [markdown]
# Most runs end at the global optimum 16 or at its neighbor 17, whose expected profit is only slightly lower, but about one in five runs ends at 13 or 14. Running the search a few times, from different initial solutions, and comparing the results (for example with [ranking and selection](lecture13_ranking-and-selection.ipynb)) makes it much more likely to find the global optimum.
#
# :::{note} Example: Shift Scheduling with Interactions
# :label: eg-8-4
#
# [Shift scheduling](lecture9_covering.ipynb#shift-scheduling) was discussed earlier. There, the required capacity per interval was given. Often, the capacity in one interval has consequences in another: customers who cannot be served now wait and need service later. Simulation is then the only way to evaluate a schedule, and the number of possible schedules is huge. Local search can be used here, with as neighbors of a schedule, for example, the schedules with one shift more or one shift less.
# :::
#
# :::{exercise}
# :label: ex-ls-neighborhood
#
# Change the function `neighbors` so that the neighborhood of $\pi$ consists of all order sizes in $S$ within distance 2 of $\pi$ (e.g., $N(5) = \{3, 4, 6, 7\}$ and $N(0) = \{1, 2\}$). Repeat the 300 runs. Does the search end in the local optimum less often? Explain why.
# :::
#
# :::{exercise}
# :label: ex-8-4
#
# Solve the [newsvendor exercise of Ranking and Selection](lecture13_ranking-and-selection.ipynb#ex-8-2) using local search with a budget of 5000. Take $N(\pi) = \{\pi-1, \pi+1\}$, with $N(1) = \{2\}$ and $N(50) = \{49\}$. Plot the current solution as the algorithm progresses.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.3.
