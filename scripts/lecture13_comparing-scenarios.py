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
# This notebook covers the simplest of the [four types of simulation optimization problems](lecture13_about-simopt.ipynb#four-types): $|S| = 2$, so we only have to decide which of two scenarios is better. We compare them with a confidence interval for the difference, and show how common random numbers give a narrower interval for the same number of runs. Common random numbers return in all following notebooks: [Ranking and Selection](lecture13_ranking-and-selection.ipynb), [Local Search](lecture13_local-search.ipynb) and [Gradient Methods](lecture13_gradient-methods.ipynb).
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
# With $|S| = 2$, one scenario gives a random performance $X$ and the other a random performance $X'$, for example the throughput of two layouts of a factory. The question is whether one scenario is better than the other, that is, whether $EX \neq EX'$.
#
# The most direct approach simulates both scenarios $n$ times, independently: runs $X_1, \dots, X_n$ of the first scenario and $X_1', \dots, X_n'$ of the second. As in [Variability (Recap)](lecture12_variability-recap.ipynb#hypothesis-testing), we write $\sigma^2(X)$ for the (unknown) variance of $X$, and $\bar X$ and $S_X^2$ for the sample average and sample variance of the runs, which estimate $EX$ and $\sigma^2(X)$; the same for $X'$. To have one notation for this section and the next, let $Y = X - X'$ and $Y_i = X_i - X_i'$, the difference of the $i$-th runs. Then $\bar Y = \bar X - \bar X'$ estimates $EY = EX - EX'$, and because the runs of the two scenarios are independent,
#
# $$
# \sigma^2(Y) = \sigma^2(X) + \sigma^2(X'),
# $$
#
# so the sample variance $S_Y^2$ of the differences is approximately $S_X^2 + S_{X'}^2$. A 95% CI for $EY$ is
#
# $$
# \left[\bar Y - 1.96 \frac{S_Y}{\sqrt n}, \ \bar Y + 1.96 \frac{S_Y}{\sqrt n}\right].
# $$
#
# The [test for two independent samples](lecture12_variability-recap.ipynb#hypothesis-testing) of $H_0: EX = EX'$ comes down to this CI: at significance level 5% (two-sided), we reject $H_0$ exactly when 0 does not lie in the CI. In that case we have evidence that one scenario is better than the other. Writing the test as a CI has an advantage: the width of the CI, $2 \times 1.96\, S_Y/\sqrt n$, shows directly how much a smarter design of the simulation gains. If the design makes $\sigma(Y)$ smaller, the CI gets narrower for the same number of runs $n$.

# %% [markdown]
# (crn)=
# ## Matched Pairs and Common Random Numbers
#
# We can often do better by comparing the two scenarios in a similar, "fair" test setting. We take matched pairs of samples $(X_i, X_i')$, and make sure that the two samples of each pair are positively correlated by simulating them under similar conditions. Then we compute the differences $Y_i = X_i - X_i'$ and make the CI for $EY$ from them exactly as above, as in the [test for matched pairs](lecture12_variability-recap.ipynb#hypothesis-testing). If $0 \notin \text{CI}$, the difference is significant.
#
# The pairing gives more statistical power because it makes the CI narrower. For $X$ and $X'$ that are not independent,
#
# $$
# \sigma^2(Y) = \sigma^2(X - X') = \sigma^2(X) + \sigma^2(X') - 2\,\mathrm{Cov}(X, X').
# $$
#
# The covariance $\mathrm{Cov}(X, X')$ measures how $X$ and $X'$ move together. If it is positive, a large sample of $X$ tends to come with a large sample of $X'$, so their difference is relatively small. So a positive covariance reduces $\sigma^2(Y)$ compared to independent samples, where the covariance is 0. A smaller $\sigma(Y)$ gives a narrower CI for the same $n$, and as a result, if $EY \neq 0$, the CI excludes 0 after fewer runs: we see earlier that one scenario is better.
#
# In practice, we get this positive correlation with **common random numbers** (CRN): we use the same underlying random draws for both scenarios (the same simulated demands, the same arrival moments, or in general the same seed), instead of drawing fresh randomness for each scenario.

# %% [markdown]
# (crn-example)=
# ## Example: The Effect of Common Random Numbers
#
# A shop decides between ordering 9 or 10 units of a product. Demand is Poisson with mean 10, a unit is sold for €5 and bought for €3, and unsold units are worthless. We simulate the difference in profit between the two order quantities 1000 times, once with independent demand for the two scenarios and once with CRN, where the same demand feeds both:

# %%
import numpy as np

rng = np.random.default_rng(1)
n_runs = 1000
price, cost = 5, 3
order_a, order_b = 9, 10


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
        f"{name:>11}: s_Y = {sd:5.2f}, 95% CI for EY: "
        f"[{mean - half_width:.2f}, {mean + half_width:.2f}]"
    )

# %% [markdown]
# With independent sampling, the CI contains 0: after 1000 runs we cannot tell which order quantity is better. With CRN, the CI lies entirely below 0, so ordering 10 units gives a lower expected profit than ordering 9. [](#fig-crn-independent) and [](#fig-crn-common) show why: with CRN, a day with high demand raises the profit of both order quantities, and a day with low demand lowers both, so the two profits move together and their difference varies much less.

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
    fig.update_yaxes(title="profit (€)", range=[-32, 22], **AXIS_STYLE)
    fig.show(config=PLOT_CONFIG)


# %% tags=["remove-cell"] label="crn-independent"
plot_traces(profit_a_indep, profit_b_indep)

# %% [markdown]
# :::{figure} #crn-independent
# :label: fig-crn-independent
#
# Profit of ordering 9 and 10 units in the first 40 runs, with independent demand for the two order quantities.
# :::

# %% tags=["remove-cell"] label="crn-common"
plot_traces(profit_a_crn, profit_b_crn)

# %% [markdown]
# :::{figure} #crn-common
# :label: fig-crn-common
#
# Profit of ordering 9 and 10 units in the first 40 runs, with common random numbers: both order quantities face the same demand in each run.
# :::
#
# How many runs does CRN save here? The CI has width $2 \times 1.96\, s_Y/\sqrt n$, where $s_Y$ is the sample standard deviation of the differences. Because of the $\sqrt n$, halving the width takes 4 times as many runs, and in general, making the width $f$ times smaller takes $f^2$ times as many runs. CRN makes $s_Y$, and thus the width, $f$ times smaller without extra runs, with
#
# $$
# f = \frac{s_Y \text{ with independent sampling}}{s_Y \text{ with CRN}}.
# $$
#
# To get the CI of CRN with independent sampling, we would need $f^2$ times as many runs:

# %%
sd_ratio = diff_indep.std(ddof=1) / diff_crn.std(ddof=1)
print(f"f = {sd_ratio:.1f}, f^2 = {sd_ratio**2:.0f}")
n_equivalent = sd_ratio**2 * n_runs
print(f"independent runs per scenario for the same width: {n_equivalent:.0f}")

# %% [markdown]
# CRN makes the CI about 4.5 times narrower, which with independent sampling would take about 20 times as many runs: about 20,500 runs per scenario instead of 1000. And CRN costs nothing extra: we only reuse the demands we drew. The two order quantities are close, so their profits are strongly correlated under CRN; for scenarios that differ more, the savings are smaller.
#
# :::{exercise}
# :label: ex-crn-savings
#
# In the example above, change `order_b` to 13.
#
# a. Before running the code: do you expect $f$ to be larger or smaller than for `order_b = 10`? Explain.
#
# b. Run the code and compute how many runs per scenario independent sampling would need for the same CI width as CRN.
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
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.1.
