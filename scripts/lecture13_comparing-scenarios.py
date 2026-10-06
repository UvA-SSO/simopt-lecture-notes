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
# description: "Comparing two scenarios by simulation: a confidence interval for the difference, matched pairs, and common random numbers."
# thumbnail: null
# ---
# # Lecture 13: Comparing Scenarios
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_comparing-scenarios.ipynb)

# %% [markdown]
# This notebook covers the simplest of the [four types of simulation optimization problems](lecture13_about-simopt.ipynb#four-types): $|S| = 2$, so we only have to decide which of two scenarios is better. We compare them with a confidence interval for the difference, and show how common random numbers give a narrower interval for the same number of runs. The same idea returns in [Ranking and Selection](lecture13_ranking-and-selection.ipynb) and [Gradient Methods](lecture13_gradient-methods.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - compare two scenarios with a confidence interval for the difference of their expected performances
# - explain why positively correlated matched pairs reduce the variance of the difference
# - use common random numbers in a simulation, and quantify how many runs they save

# %% [markdown]
# (comparing-scenarios)=
# ## Independent Samples
#
# With $|S| = 2$, one scenario gives a random performance $X$ and the other a random performance $X'$, for example the throughput of two layouts of a factory. The question is whether one scenario is better than the other, that is, whether $EX \neq EX'$. The most direct approach simulates $X$ and $X'$ independently, with an equal number of runs $n$ of both scenarios, and applies the [test for two independent samples](lecture12_variability-recap.ipynb#hypothesis-testing): make a CI for $EX - EX'$, and if this CI does not contain 0, we have evidence that one scenario is better than the other.

# %% [markdown]
# (crn)=
# ## Matched Pairs and Common Random Numbers
#
# We can often do better by designing the simulation experiment more cleverly, which gives more statistical power for the same number of runs. The idea is to compare the two scenarios in a similar, "fair" test setting. We take matched pairs of samples $(X_i, X_i')$, and make sure that the two samples of each pair are positively correlated by simulating them under similar conditions. Then $Y = X - X'$ has $EY = EX - EX'$, and instead of a two-sample test we compute the differences $Y_i = X_i - X_i'$ and make a CI for $EY$ from them, as in the [test for matched pairs](lecture12_variability-recap.ipynb#hypothesis-testing). If $0 \notin \text{CI}$, the difference is significant.
#
# This works because
#
# $$
# \sigma^2(Y) = \sigma^2(X - X') = \sigma^2(X) + \sigma^2(X') - 2\,\mathrm{Cov}(X, X').
# $$
#
# The covariance $\mathrm{Cov}(X, X')$ measures how $X$ and $X'$ move together. If it is positive, a large sample of $X$ tends to come with a large sample of $X'$, so their difference is relatively small. For independent samples the covariance is 0, so making $\mathrm{Cov}(X, X') > 0$ shrinks $\sigma^2(Y)$, and a smaller $\sigma^2(Y)$ gives a narrower CI for the same number of runs.
#
# In practice, we get this positive correlation with **common random numbers** (CRN): we use the same underlying random draws for both scenarios (the same simulated demands, the same arrival moments, or in general the same seed), instead of drawing fresh randomness for each scenario.

# %% [markdown]
# (crn-example)=
# ## Worked Example: Two Order Quantities
#
# A shop decides between ordering 8 or 12 units of a product. Demand is Poisson with mean 10, a unit is sold for €5 and bought for €3, and unsold units are worthless. We simulate the difference in profit between the two order quantities 5000 times, once with independent demand for the two scenarios and once with CRN, where the same demand feeds both:

# %%
import numpy as np

rng = np.random.default_rng(7)
n_runs = 5000
price, cost = 5, 3
order_a, order_b = 8, 12


def profit(order, demand):
    return price * np.minimum(demand, order) - cost * order


# independent sampling: a separate demand stream per scenario
profit_a_indep = profit(order_a, rng.poisson(10, n_runs))
profit_b_indep = profit(order_b, rng.poisson(10, n_runs))
diff_indep = profit_b_indep - profit_a_indep

# common random numbers: one shared demand stream for both scenarios
demand_shared = rng.poisson(10, n_runs)
profit_a_crn = profit(order_a, demand_shared)
profit_b_crn = profit(order_b, demand_shared)
diff_crn = profit_b_crn - profit_a_crn

for name, diff in [("independent", diff_indep), ("CRN", diff_crn)]:
    mean, sd = diff.mean(), diff.std(ddof=1)
    half_width = 1.96 * sd / np.sqrt(n_runs)
    print(
        f"{name:>11}: Var(diff) = {sd**2:6.1f}, 95% CI for the "
        f"difference: [{mean - half_width:.2f}, {mean + half_width:.2f}]"
    )

# %% [markdown]
# Both CIs lie entirely below 0, so ordering 12 units gives a lower expected profit than ordering 8. The CI with CRN is clearly narrower. [](#fig-crn-independent) and [](#fig-crn-common) show why: with CRN, a day with high demand raises the profit of both order quantities, and a day with low demand lowers both, so the two profits move together and their difference varies less.

# %% tags=["remove-cell"]
import plotly.graph_objects as go

TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
    "height": 320,
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.25,
    },
}
AXIS_STYLE = {"gridcolor": "rgba(128,128,128,0.25)"}
# static pictures: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}


def plot_traces(profit_a, profit_b, n_shown=40):
    fig = go.Figure()
    runs = np.arange(1, n_shown + 1)
    for values, order, color in [
        (profit_a, order_a, "#1f77b4"),
        (profit_b, order_b, "#ff7f0e"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=runs,
                y=values[:n_shown],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5},
                name=f"order {order}",
            )
        )
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="simulation run", **AXIS_STYLE)
    fig.update_yaxes(title="profit (€)", range=[-25, 30], **AXIS_STYLE)
    fig.show(config=PLOT_CONFIG)


# %% tags=["remove-cell"] label="crn-independent"
plot_traces(profit_a_indep, profit_b_indep)

# %% [markdown]
# :::{figure} #crn-independent
# :label: fig-crn-independent
#
# Profit of ordering 8 and 12 units in the first 40 runs, with independent demand for the two order quantities.
# :::

# %% tags=["remove-cell"] label="crn-common"
plot_traces(profit_a_crn, profit_b_crn)

# %% [markdown]
# :::{figure} #crn-common
# :label: fig-crn-common
#
# Profit of ordering 8 and 12 units in the first 40 runs, with common random numbers: both order quantities face the same demand in each run.
# :::
#
# How many runs does CRN save here? The width of the CI is proportional to $\sigma(Y)/\sqrt{n}$. With independent sampling, the variance of the difference is about 2.4 times as large as with CRN, so independent sampling needs about 2.4 times as many runs for a CI of the same width:

# %%
variance_ratio = diff_indep.var(ddof=1) / diff_crn.var(ddof=1)
n_equivalent = variance_ratio * n_runs
print(f"variance ratio: {variance_ratio:.2f}")
print(f"independent pairs for the same CI width: {n_equivalent:.0f}")

# %% [markdown]
# With CRN, 5000 pairs (10,000 simulation runs) give a CI that independent sampling only reaches with about 12,000 pairs (24,100 runs). CRN saves about 14,100 runs here, more than half of the work, and it costs nothing extra: we only reuse the demands we drew. In more complex simulations, where the two scenarios differ less, the correlation and thus the savings are often much larger.
#
# :::{exercise}
# :label: ex-crn-savings
#
# In the example above, change `order_b` to 9.
#
# a. Before running the code: do you expect the variance ratio to be larger or smaller than for `order_b = 12`? Explain.
#
# b. Run the code and compute how many simulation runs CRN saves now.
# :::
#
# :::{exercise}
# :label: ex-8-1
#
# We extend the [budget exercise](lecture12_monte-carlo.ipynb#ex-5-3) with a second product. Management has to decide between both products. All variables are again normally distributed, with mean and SD 12000 and 2000 for sales; 90 and 20 for the price; 68 and 20 for the variable costs; 165000 and 30000 for the fixed costs.
#
# a. Simulate both scenarios 10000 times and make a CI for the difference. Which one is better? Explain your answer.
#
# b. Do the same, but using common random numbers. What is the difference?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.1, "Comparing Scenarios."
