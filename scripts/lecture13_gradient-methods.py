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
# description: "Gradient methods for simulation optimization over a continuous set of solutions: finite-difference gradient estimates, step sizes, and common random numbers."
# thumbnail: null
# ---
# # Lecture 13: Gradient Methods
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_gradient-methods.ipynb)

# %% [markdown]
# The last of the [four types of problems](lecture13_about-simopt.ipynb#four-types) has a continuous set of solutions $S$, such as an interval. [Local search](lecture13_local-search.ipynb) moves between neighbors, but in a continuous set there is no natural neighbor. Instead, we take continuous steps in the direction in which the expected performance increases fastest, estimated by simulation. This notebook applies such a gradient method to a newsvendor that sells milk by the liter.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - describe the gradient ascent algorithm and the role of the step size
# - estimate a gradient by simulation with finite differences
# - explain why common random numbers are needed for a useful finite-difference estimate
# - explain the downside of common random numbers: less noise also means less exploration

# %% [markdown]
# (gradient-methods)=
# ## Gradient Ascent
#
# When $S$ is continuous, for example the number of seconds that a traffic light stays red, we use the gradient $\nabla E[r(X, \pi)]$ of the expected performance. For a single decision variable, the gradient is the derivative: the slope of $E[r(X, \pi)]$ at $\pi$. For several decision variables, it is the vector of partial derivatives, and it points in the direction in which $E[r(X, \pi)]$ increases fastest. We maximize, so we step from the current solution $\pi_k$ in the direction of the gradient, uphill:
#
# $$
# \pi_{k+1} = \pi_k + \gamma_{k+1} \nabla E[r(X, \pi_k)].
# $$
#
# If the slope at $\pi_k$ is positive, $\pi_{k+1}$ is larger than $\pi_k$; if it is negative, smaller. The **step size** $\gamma_{k+1} > 0$ determines how far we move. Near a maximum the slope is close to 0, so the steps automatically get smaller. Theory recommends $\gamma_k = 1/k$ (or a multiple of it): the steps shrink, and under fairly general conditions the method converges. In practice, the step size is often tuned to the problem, because $1/k$ can become very small long before a good solution is reached. Because we maximize, this method is called gradient ascent; for minimization we step in the opposite direction, with a minus sign, and it is called gradient descent.
#
# (finite-differences)=
# ## Estimating the Gradient
#
# As everything else in this lecture, $E[r(X, \pi)]$ can only be simulated, so we cannot compute its gradient: we estimate it. The **finite-difference method** simulates $n$ runs at $\pi_k$ and $n$ runs at $\pi_k + \delta$, for a small $\delta > 0$, and uses the slope between the two averages:
#
# $$
# \nabla E[r(X, \pi_k)] \approx \frac{y(\pi_k + \delta) - y(\pi_k)}{\delta}.
# $$
#
# With more decision variables, we estimate each partial derivative in this way, by changing one variable at a time. A side effect is that every iteration also gives an estimate $y(\pi_k)$ of the performance of the current solution.
#
# Here $\pi_k$ and $\pi_k + \delta$ are two very similar scenarios, so [common random numbers](lecture13_comparing-scenarios.ipynb#crn) make the difference $y(\pi_k + \delta) - y(\pi_k)$ much less noisy, as in Comparing Scenarios. This matters even more than there, because the difference is divided by the small number $\delta$, which magnifies its noise.

# %% [markdown]
# (milk-example)=
# ## Example: A Newsvendor for Milk
#
# A shop sells fresh milk, which it orders every morning: $\pi$ liters, anywhere in $S = [0, 30]$. Milk costs €0.60 per liter and sells for €1 per liter, and milk that is left at the end of the day is thrown away. The demand $X$ is continuous: Gamma distributed with shape 10 and scale 1.5, so the mean demand is 15 liters. The profit is $r(X, \pi) = \min(X, \pi) - 0.6\pi$, as for the [newsvendor](lecture13_ranking-and-selection.ipynb#newsvendor), but now $\pi$ is continuous.
#
# Each iteration simulates $n = 100$ days at $\pi_k$ and at $\pi_k + \delta$ with $\delta = 0.1$, starting at $\pi_0 = 2$ with step size $\gamma_k = 20/k$. The function `gradient_ascent` runs the method, with or without common random numbers; with `crn=True`, both order sizes face the same 100 simulated demands. As for the algorithms of Lecture 11, you should be able to read and follow this code; you do not need to be able to write it from scratch.

# %%
import numpy as np

price, cost = 1, 0.6
shape, scale = 10, 1.5
lower, upper = 0, 30


def profit(order, demand):
    return price * np.minimum(demand, order) - cost * order


def gradient_ascent(start, n_iter, n, delta, crn, rng):
    order = start
    path = [order]
    estimates = []  # y(pi_k) of every iteration
    for k in range(1, n_iter + 1):
        demand = rng.gamma(shape, scale, n)
        if crn:
            demand_plus = demand
        else:
            demand_plus = rng.gamma(shape, scale, n)
        y_order = profit(order, demand).mean()
        y_plus = profit(order + delta, demand_plus).mean()
        gradient = (y_plus - y_order) / delta
        estimates.append(y_order)
        step_size = 20 / k
        # take the step, but stay inside S = [lower, upper]
        order = min(max(order + step_size * gradient, lower), upper)
        path.append(order)
    return path, estimates


rng = np.random.default_rng(1)
path_crn, estimates_crn = gradient_ascent(2, 100, 100, 0.1, True, rng)
for k in range(5):
    print(
        f"pi_{k} = {path_crn[k]:5.2f}: y(pi_{k}) = {estimates_crn[k]:.2f}, "
        f"pi_{k + 1} = {path_crn[k + 1]:.2f}"
    )
print(f"pi_100 = {path_crn[-1]:.2f}")

# %% [markdown]
# At $\pi_0 = 2$ the demand is almost always larger than the order, so one extra liter is sold for €1 and costs €0.60: the gradient estimate is 0.4, and the first step takes us from 2 to $2 + 20 \times 0.4 = 10$. Closer to the mean demand, an extra liter is sold less often, the gradient gets smaller, and so do the steps.
#
# For this simple example, the optimal order size can be calculated exactly: $\pi^* \approx 13.36$ liters (how is outside the scope of this course; the code is hidden below). For real-life problems this is in general not possible, and we only use it to check the method. We now run the method once more without common random numbers:

# %% tags=["hide-input"]
# exact optimum, only to check the method (outside the course's scope)
from scipy import stats

optimum = stats.gamma.ppf((price - cost) / price, shape, scale=scale)


# %%
rng = np.random.default_rng(1)
path_indep, estimates_indep = gradient_ascent(2, 100, 100, 0.1, False, rng)
print(f"with CRN, pi_100 = {path_crn[-1]:.2f}")
print(f"without CRN, pi_100 = {path_indep[-1]:.2f}")

# %% tags=["remove-cell"] label="gradient-paths"
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
RUNS = [
    (path_crn, "#1f77b4", "with CRN"),
    (path_indep, "#ff7f0e", "without CRN"),
]

paths_fig = go.Figure()
for path, color, name in RUNS:
    paths_fig.add_trace(
        go.Scatter(
            x=np.arange(len(path)),
            y=path,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 4},
            name=name,
        )
    )
paths_fig.add_trace(
    go.Scatter(
        x=[0, len(path_crn) - 1],
        y=[optimum, optimum],
        mode="lines",
        line={"color": "grey", "dash": "dash", "width": 2},
        name="optimum π*",
    )
)
paths_fig.update_layout(**PLOT_LAYOUT)
paths_fig.update_xaxes(title="iteration k", **AXIS_STYLE)
paths_fig.update_yaxes(
    title="order size πₖ (liters)", range=[0, 30], **AXIS_STYLE
)
paths_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #gradient-paths
# :label: fig-gradient-paths
#
# Order size $\pi_k$ during 100 iterations of gradient ascent, with and without common random numbers, and the optimum $\pi^*$.
# :::
#
# [](#fig-gradient-paths) shows the difference. With CRN, the method approaches the optimum in a few iterations and then stays close to it. Without CRN, the gradient estimates are so noisy that the order size jumps around: the second step even takes it back to 0 liters, and after 100 iterations it is still almost a liter below the optimum.

# %% [markdown]
# (gradient-local-optima)=
# ### Local Optima
#
# Like local search, a gradient method only finds a local optimum: it follows the slope uphill and stops where the slope is 0. If $E[r(X, \pi)]$ has several local optima, the result depends on the starting point.
#
# Here common random numbers have a downside. With CRN, the gradient estimates are accurate, and the method behaves almost like deterministic gradient ascent: it climbs to the nearest local optimum and stays there. Without CRN, the noisy estimates sometimes cause large random steps, and such a step can carry the method past a dip to a better optimum, just as the noise in [stochastic local search](lecture13_local-search.ipynb#local-search) leads to more exploration. In the volume-discount version of the milk example ([](#ex-gradient-discount)), runs without CRN regularly end beyond the discount, while runs with CRN that start below it never do. But this exploration is not controlled: the same noise makes the method less precise near an optimum, as [](#fig-gradient-paths) shows. So CRN is a choice between precision and exploration. If we use CRN, we have to add exploration on purpose, for example by running the method from several starting points, and how well that works depends on the problem.
#
# The volume discount also shows a limit of gradient methods in general: the gradient is a local slope, so it cannot see a jump in $E[r(X, \pi)]$ such as the one at 15 liters. For such problems, a discrete search over candidate solutions, with [local search](lecture13_local-search.ipynb) or [ranking and selection](lecture13_ranking-and-selection.ipynb), is a better fit.
#
# :::{exercise}
# :label: ex-gradient-delta
#
# Run `gradient_ascent` with and without common random numbers for $\delta = 0.01$ and for $\delta = 1$.
#
# a. How does $\delta$ affect the run without common random numbers? Explain this with the finite-difference formula for the gradient.
#
# b. Why does a small $\delta$ hardly matter for the run with common random numbers?
# :::
#
# :::{exercise}
# :label: ex-gradient-discount
#
# Give the milk shop the volume discount of the [local search example](lecture13_local-search.ipynb#local-search-example): milk costs €0.50 per liter if the shop orders more than 15 liters.
#
# a. Change `profit` accordingly and run gradient ascent with common random numbers from $\pi_0 = 2$ and from $\pi_0 = 25$. Explain the results.
#
# b. Run it from $\pi_0 = 2$ without common random numbers, for ten different seeds. How many runs end above 15 liters? What does this say about the choice between precision and exploration?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 8, "Simulation Optimization."
# - Fu, M.C. (2002). "Optimization for simulation: Theory vs. practice." *INFORMS Journal on Computing*, 14:192–215.
