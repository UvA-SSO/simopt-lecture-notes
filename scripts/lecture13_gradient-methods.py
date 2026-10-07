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
# \nabla E[r(X, \pi_k)] \approx \hat g_k = \frac{y(\pi_k + \delta) - y(\pi_k)}{\delta}.
# $$
#
# With more decision variables, we estimate each partial derivative in this way, by changing one variable at a time. A side effect is that every iteration also gives an estimate $y(\pi_k)$ of the performance of the current solution.
#
# The choice of $\delta$ is a trade-off. A large $\delta$ gives the slope over a long interval instead of at $\pi_k$, so the estimate is biased. A small $\delta$ makes this bias small, but makes the estimate noisy. To see how noisy, compare the variance of $\hat g_k$ in two cases (Glasserman, 2004, §7.1; Asmussen & Glynn, 2007, Chapter VII):
#
# - With independent runs at $\pi_k$ and $\pi_k + \delta$, the two averages are independent, so
#
#   $$
#   \sigma^2(\hat g_k) = \frac{\sigma^2(r(X, \pi_k + \delta)) + \sigma^2(r(X, \pi_k))}{n \delta^2} \approx \frac{2\sigma^2(r(X, \pi_k))}{n \delta^2}.
#   $$
#
#   The variance grows like $1/\delta^2$: halving $\delta$ makes it four times as large, and for a small $\delta$ the estimate is mostly noise.
#
# - With [common random numbers](lecture13_comparing-scenarios.ipynb#crn), both averages use the same random draws, so $\hat g_k$ is the average of $n$ differences $\big(r(X_i, \pi_k + \delta) - r(X_i, \pi_k)\big)/\delta$, and
#
#   $$
#   \sigma^2(\hat g_k) = \frac{\sigma^2\big(r(X, \pi_k + \delta) - r(X, \pi_k)\big)}{n \delta^2}.
#   $$
#
#   If $r(x, \pi)$ changes at most proportionally to the change in $\pi$, that is, $|r(x, \pi + \delta) - r(x, \pi)| \le L\delta$ for some constant $L$ and every $x$, then each difference divided by $\delta$ lies between $-L$ and $L$, and $\sigma^2(\hat g_k) \le L^2/n$, whatever $\delta$ is. For the newsvendor below this holds with $L = 1$: ordering $\delta$ liters more changes the profit of a day by at most $\delta$ euros.
#
# So $\pi_k$ and $\pi_k + \delta$ are two very similar scenarios, and simulating them with the same random draws lets most of the noise cancel in their difference. Without CRN, finite differences only work with a large $\delta$ or a very large $n$.

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
# exact optimum and expected profit, only to check the method (outside the
# scope of the course)
from scipy import stats

optimum = stats.gamma.ppf((price - cost) / price, shape, scale=scale)
grid = np.linspace(lower, upper, 301)
expected_sales = scale * shape * stats.gamma.cdf(
    grid, shape + 1, scale=scale
) + grid * stats.gamma.sf(grid, shape, scale=scale)
expected_profit = price * expected_sales - cost * grid


def exact_profit(order):
    return np.interp(order, grid, expected_profit)


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
# [](#fig-gradient-paths) shows the difference. With CRN, the method approaches the optimum in a few iterations and then stays close to it. Without CRN, the gradient estimates are so noisy that the order size jumps around: the second step even takes it back to 0 liters, and after 100 iterations it is still almost a liter below the optimum. [](#fig-gradient-steps-crn) and [](#fig-gradient-steps-indep) show the first five steps of both runs on the expected profit curve.

# %% tags=["remove-cell"]
SUBSCRIPTS = "₀₁₂₃₄₅"


def plot_steps(path, color, positions):
    """The first order sizes of a run on the expected profit curve."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=expected_profit,
            mode="lines",
            line={"color": "grey", "width": 2},
            name="expected profit",
        )
    )
    shown = np.array(path[: len(positions)])
    fig.add_trace(
        go.Scatter(
            x=shown,
            y=exact_profit(shown),
            mode="markers+text",
            marker={"color": color, "size": 9},
            text=[f"π{SUBSCRIPTS[k]}" for k in range(len(shown))],
            textposition=positions,
            name="order sizes π₀, ..., π₅",
        )
    )
    fig.update_layout(**{**PLOT_LAYOUT, "height": 300})
    fig.update_xaxes(
        title="order size π (liters)", range=[-1, 20], **AXIS_STYLE
    )
    fig.update_yaxes(
        title="expected profit (€)", range=[-0.8, 5], **AXIS_STYLE
    )
    fig.show(config=PLOT_CONFIG)


# %% tags=["remove-cell"] label="gradient-steps-crn"
plot_steps(
    path_crn,
    "#1f77b4",
    [
        "bottom right",
        "top center",
        "top left",
        "top right",
        "bottom left",
        "bottom right",
    ],
)

# %% [markdown]
# :::{figure} #gradient-steps-crn
# :label: fig-gradient-steps-crn
#
# The first five steps of gradient ascent with common random numbers, on the expected profit curve: the order sizes $\pi_0, \dots, \pi_5$.
# :::

# %% tags=["remove-cell"] label="gradient-steps-indep"
plot_steps(
    path_indep,
    "#ff7f0e",
    [
        "bottom left",
        "top center",
        "top center",
        "top right",
        "top left",
        "top left",
    ],
)

# %% [markdown]
# :::{figure} #gradient-steps-indep
# :label: fig-gradient-steps-indep
#
# The first five steps of gradient ascent without common random numbers, on the expected profit curve: the order sizes $\pi_0, \dots, \pi_5$.
# :::
#
# With CRN, the run climbs the curve from $\pi_0 = 2$ via $\pi_1 = 10$ to the top, and then makes small steps around it. Without CRN, the first step is the same (at $\pi_0 = 2$ the profit hardly varies, so there is little noise), but the second gradient estimate is so far off that the run falls back to 0, from where it climbs again.
#
# The averages $y(\pi_k)$ are based on only 100 runs each, so they tell us little about how good the final solution is, and the method itself never simulates $\pi_{100}$. To report the expected profit of the final solution, we therefore simulate it separately, with new, independent runs:

# %%
final_order = path_crn[-1]
final_profits = profit(final_order, rng.gamma(shape, scale, 10000))
mean, sd = final_profits.mean(), final_profits.std(ddof=1)
half_width = 1.96 * sd / np.sqrt(len(final_profits))
print(f"final order size: {final_order:.2f} liters")
print(
    f"95% CI for its expected profit: [{mean - half_width:.2f}, "
    f"{mean + half_width:.2f}]"
)

# %% [markdown]
# Using new runs here, instead of the runs that the method used along the way, avoids the overestimation described in [About Simulation Optimization](lecture13_about-simopt.ipynb#simopt-challenges): the final solution was chosen because its estimates looked good, so those same estimates tend to be too high.
#
# (gradient-local-optima)=
# ### Local Optima
#
# Like local search, a gradient method only finds a local optimum: it follows the slope uphill and stops where the slope is 0. If $E[r(X, \pi)]$ has several local optima, the result depends on the starting point. The noise of an estimate without CRN sometimes carries the method past a dip to a better optimum (see [](#ex-gradient-discount)), just as the noise in [stochastic local search](lecture13_local-search.ipynb#local-search) can. But that is luck: the same noise also throws the method away from a good solution, and where it ends is hard to predict. A more reliable way to look for a better optimum is to estimate the gradient accurately, with CRN, and to run the method from several starting points.
#
# :::{exercise}
# :label: ex-gradient-delta
#
# Run `gradient_ascent` with and without common random numbers for $\delta = 0.01$ and for $\delta = 1$.
#
# a. How does $\delta$ affect the run without common random numbers? Explain with the variance of the gradient estimate.
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
# b. Run it from $\pi_0 = 2$ without common random numbers, for ten different seeds. How many runs end above 15 liters? Is this a good way to escape a local optimum?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 8, "Simulation Optimization."
# - Asmussen, S. & Glynn, P.W. (2007). *Stochastic Simulation: Algorithms and Analysis*. Springer. Chapter VII, "Derivative Estimation."
# - Fu, M.C. (2002). "Optimization for simulation: Theory vs. practice." *INFORMS Journal on Computing*, 14:192–215.
# - Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer. §7.1, "Finite-Difference Approximations."
