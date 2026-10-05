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
# description: "Monte Carlo simulation with numpy arrays: estimating an expectation or a probability with a confidence interval, applied to project planning."
# thumbnail: null
# ---
# # Lecture 12: Monte Carlo Simulation
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture12_monte-carlo.ipynb)

# %% [markdown]
# Monte Carlo simulation estimates the expected output $E[r(X)]$ of a model $r$ with random inputs $X$ by averaging the outputs of many samples. It applies when $r$ is a known function of a fixed number of random inputs, as in the project of [Why Simulate?](lecture12_why-simulation.ipynb#flaw-of-averages). This notebook explains why the method works and how accurate it is, shows how to simulate with `numpy` arrays instead of loops, and then works out the project planning example. It uses the sampling methods from [Sampling a Random Variable](lecture12_sampling.ipynb). Processes that are too complex for a single function $r$ follow in [Discrete-Event Simulation](lecture12_discrete-event-simulation.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - explain why the average of simulated outputs converges to $E[r(X)]$
# - compute a confidence interval for an expectation or a probability from simulation output
# - use `numpy` arrays to perform a Monte Carlo simulation, and read such code
# - simulate the finish time of a project with random activity durations

# %% [markdown]
# (the-method)=
# ## The Method
#
# Monte Carlo simulation approximates $E[r(X)]$ in three steps:
#
# 1. sample independent realizations $x_1, \dots, x_n$ of $X$;
# 2. compute the outputs $r(x_1), \dots, r(x_n)$;
# 3. average them: $\big(r(x_1) + \dots + r(x_n)\big)/n$.
#
# **Why does this converge?** Write $Y = r(X)$ and $Y_i = r(X_i)$ for the output of run $i$, so $Y_1, \dots, Y_n$ are independent with the same distribution as $Y$. By the [rules for sums and averages](lecture12_variability-recap.ipynb#sums-of-rvs),
#
# $$
# E\left[\frac{Y_1 + \dots + Y_n}{n}\right] = EY, \qquad
# \sigma^2\left[\frac{Y_1 + \dots + Y_n}{n}\right] = \frac{\sigma^2(Y)}{n}.
# $$
#
# The average is centered on $EY$, and its variance goes to 0 as $n$ grows. This is the [law of large numbers](lecture12_variability-recap.ipynb#lln): the average output of the simulation converges to the expected output.
#
# **How accurate is the estimate?** With $n$ runs, sample mean $m$ and sample SD $s$ of the outputs, a 95% [confidence interval](lecture12_variability-recap.ipynb#confidence-intervals) for $E[r(X)]$ is
#
# $$
# \left[m - \frac{2s}{\sqrt n}, \ m + \frac{2s}{\sqrt n}\right].
# $$
#
# The width shrinks with $\sqrt n$: twice as precise requires four times as many runs.
#
# **Probabilities.** Sometimes we want a probability such as $P(r(X) > \alpha)$ instead of an expectation. A probability is the expectation of an indicator: $I\{r(X) > \alpha\}$ is 1 if $r(X) > \alpha$ and 0 otherwise, and
#
# $$
# E[I\{r(X) > \alpha\}] = 1 \cdot P(r(X) > \alpha) + 0 \cdot P(r(X) \le \alpha) = P(r(X) > \alpha).
# $$
#
# So we simulate the 0/1 outputs $I\{r(x_i) > \alpha\}$: their average is the fraction of runs with $r(x_i) > \alpha$, and the same formula gives a CI.

# %% [markdown]
# (numpy-arrays)=
# ## Simulating with numpy Arrays
#
# A simulation can loop over the runs, but in Python it pays to use `numpy` wherever possible: it is faster and shorter to sample every random input as an array with one value per run, and to compute with whole arrays. A Python `for` loop handles one number per step, and each step has overhead: Python checks the type of every number and looks up every operation again. A `numpy` operation on an array is a single call to compiled code that processes all values in a row.
#
# As an example, we estimate the probability that the sum of two dice is 7, which we know is $1/6 \approx 0.167$, once with a loop and once with arrays:

# %%
from time import perf_counter

import numpy as np

rng = np.random.default_rng(1)
n_runs = 100000

start_time = perf_counter()
sevens = 0
for _ in range(n_runs):
    dice_sum = rng.integers(1, 7) + rng.integers(1, 7)
    if dice_sum == 7:
        sevens += 1
loop_seconds = perf_counter() - start_time

start_time = perf_counter()
dice_sums = rng.integers(1, 7, n_runs) + rng.integers(1, 7, n_runs)
array_fraction = (dice_sums == 7).mean()
array_seconds = perf_counter() - start_time

print(f"loop:   {loop_seconds:.3f} s, fraction {sevens / n_runs:.4f}")
print(f"arrays: {array_seconds:.4f} s, fraction {array_fraction:.4f}")
print(f"the arrays are {loop_seconds / array_seconds:.0f} times faster")

# %% [markdown]
# Both versions give about 1/6, but the array version is many times faster. That matters once a simulation is repeated many times, as in Lecture 13, where a whole simulation is run for every decision we want to compare.
#
# Below are the `numpy` features that are useful to perform a Monte Carlo simulation, grouped by the three steps of [the method](#the-method).
#
# **Sampling the inputs.** Every sampling method of a random generator returns an array of $n$ values, one per run, when it gets the number of samples $n$ as its last argument, for example `rng.uniform(a, b, n)` or `rng.lognormal(mu, sigma, n)`. [Sampling a Random Variable](lecture12_sampling.ipynb#sampling-directly) lists these methods. With several random inputs, a dictionary with one array per input keeps them together, for example one array of durations per activity of a project: `durations = {"A": rng.uniform(2, 4, n), "B": ...}`.
#
# **Evaluating the function $r$.** Operations on arrays of the same length work run by run, so the code for $r$ looks like the formula for a single run:
#
# | what | code | result |
# |---|---|---|
# | arithmetic per run | `x + y`, `price * sales - costs`, `x ** 2` | array, computed value by value |
# | maximum or minimum per run | `np.maximum(x, y)`, `np.minimum(x, 45)` | array; a number such as 45 is used for every run |
# | a condition per run | `x > 25`, `(x > 25) & (y < 3)` | array of `True`/`False` |
#
# **Analyzing the output.** These turn the array of outputs into one number:
#
# | what | code | result |
# |---|---|---|
# | summary statistics | `x.mean()`, `x.std(ddof=1)` | `ddof=1` divides by $n - 1$, giving the sample SD $s$ used in a CI |
# | a probability | `(x > 25).mean()` | fraction of runs in which the condition holds (`True` counts as 1) |
# | the largest value | `x.max()` or `np.max(x)` | the largest value in the whole array |
#
# Note the difference between `np.maximum(x, y)`, which compares two arrays run by run, and `np.max(x)`, which gives the largest value of one array.

# %% [markdown]
# (project-planning-example)=
# ## Example: Project Planning
#
# We return to the project of [Why Simulate?](lecture12_why-simulation.ipynb#flaw-of-averages): activities A and B run at the same time, and C starts when both are finished. The durations $X_A$, $X_B$ and $X_C$ (in days) are independent and lognormal with parameters $\mu = 1$ and $\sigma = 1$, which gives each activity an expected duration of $e^{1.5} \approx 4.48$ days. The finish time is
#
# $$
# r(X_A, X_B, X_C) = \max(X_A, X_B) + X_C,
# $$
#
# and plugging in the expected durations gives $4.48 + 4.48 = 8.96$ days. We simulate the project 10,000 times, with one array of durations per activity:

# %%
rng = np.random.default_rng(2026)
n_runs = 10000
durations = {
    activity: rng.lognormal(1, 1, n_runs) for activity in ["A", "B", "C"]
}
finish_time = np.maximum(durations["A"], durations["B"]) + durations["C"]
print("first five finish times:", finish_time[:5].round(2))
print("plug-in estimate r(EX):", 2 * np.exp(1.5))
print("simulated estimate of E[r(X)]:", finish_time.mean())

# %% [markdown]
# The expected finish time is more than two days later than the plug-in estimate: $E[r(X)]$ is larger than $r(EX)$, the flaw of averages, equation [](lecture12_why-simulation.ipynb#eq-flaw-of-averages) in [Why Simulate?](lecture12_why-simulation.ipynb#flaw-of-averages). [](#fig-finish-histogram) shows the distribution of the simulated finish times. It is skewed to the right: most projects finish within 15 days, but a few take much longer, and these long runs pull the expected finish time up.

# %% tags=["remove-cell"] label="finish-histogram"
import plotly.graph_objects as go

TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
    "height": 360,
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.2,
    },
}
AXIS_STYLE = {"gridcolor": "rgba(128,128,128,0.25)"}
# static pictures: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}

histogram_fig = go.Figure()
histogram_fig.add_trace(
    go.Histogram(
        x=finish_time,
        xbins={"start": 0, "end": 40, "size": 1},
        marker={"color": "rgba(31,119,180,0.6)", "line": {"width": 0}},
        name="simulated finish times",
    )
)
for value, color, dash, name in [
    (2 * np.exp(1.5), "grey", "dash", "plug-in estimate r(EX)"),
    (finish_time.mean(), "#d62728", "solid", "simulated mean"),
]:
    histogram_fig.add_trace(
        go.Scatter(
            x=[value, value],
            y=[0, 900],
            mode="lines",
            line={"color": color, "dash": dash, "width": 3},
            name=name,
        )
    )
histogram_fig.update_layout(**PLOT_LAYOUT, bargap=0.05)
histogram_fig.update_xaxes(
    title="finish time (days)", range=[0, 40], **AXIS_STYLE
)
histogram_fig.update_yaxes(
    title="number of runs", range=[0, 900], **AXIS_STYLE
)
histogram_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #finish-histogram
# :label: fig-finish-histogram
#
# Histogram of the 10,000 simulated finish times, with the plug-in estimate of 8.96 days and the simulated mean. The few runs longer than 40 days are not shown.
# :::
#
# [](#fig-running-average) shows the average finish time after the first $n$ runs, with $n$ on a logarithmic scale so that both the first runs and the later ones are visible. The first runs move the average a lot, but it settles down as $n$ grows, as the law of large numbers predicts. The jumps come from rare, very long durations, typical of the right-skewed lognormal distribution.

# %% label="running-average" tags=["remove-cell"]
runs = np.arange(1, n_runs + 1)
running_average = finish_time.cumsum() / runs
running_fig = go.Figure()
running_fig.add_trace(
    go.Scatter(
        x=runs,
        y=running_average,
        mode="lines",
        line={"color": "#1f77b4", "width": 2},
        name="average of the first n runs",
    )
)
running_fig.add_trace(
    go.Scatter(
        x=[1, n_runs],
        y=[2 * np.exp(1.5)] * 2,
        mode="lines",
        line={"color": "grey", "dash": "dash", "width": 2},
        name="plug-in estimate r(EX)",
    )
)
running_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": TEXT_COLOR, "size": 13},
    margin={"l": 50, "r": 20, "t": 20, "b": 50},
    height=360,
    legend={
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.2,
    },
)
running_fig.update_xaxes(
    title="number of runs n",
    type="log",
    gridcolor="rgba(128,128,128,0.25)",
)
running_fig.update_yaxes(
    title="finish time (days)",
    range=[0, 20],
    gridcolor="rgba(128,128,128,0.25)",
)
running_fig.show(config={"displayModeBar": False, "staticPlot": True})

# %% [markdown]
# :::{figure} #running-average
# :label: fig-running-average
#
# Average finish time of the first $n$ simulated projects, with the plug-in estimate of 8.96 days. The horizontal axis has a logarithmic scale.
# :::
#
# The 95% CI quantifies how accurate the estimate is:

# %%
mean, sd = finish_time.mean(), finish_time.std(ddof=1)
half_width = 2 * sd / np.sqrt(n_runs)
print(f"95% CI: [{mean - half_width:.2f}, {mean + half_width:.2f}]")

# %% [markdown]
# The plug-in estimate of 8.96 days is far outside it. The probability that the project takes more than 15 days follows from the same code, applied to a `True`/`False` array:

# %%
late = finish_time > 15
mean, sd = late.mean(), late.std(ddof=1)
half_width = 2 * sd / np.sqrt(n_runs)
print(f"P(finish time > 15): {mean:.3f}")
print(f"95% CI: [{mean - half_width:.3f}, {mean + half_width:.3f}]")

# %% [markdown]
# With the expected durations, the project would never take more than 15 days, but in about one in five simulated projects it does.
#
# :::{exercise}
# :label: ex-5-1
#
# Take the [project planning problem](lecture8_linear-optimization.ipynb#project-planning) from Lecture 8, where the durations in the table are now the expected durations of the activities.
#
# a. Write the finish time of the project as a function $r$ of the six durations, using $\max$ and $+$.
#
# b. Simulate the project with every duration uniform on $[d - 1, d + 1]$, where $d$ is the expected duration. Give a 95% CI for the expected finish time, and compare it with the finish time for the expected durations.
#
# c. Simulate the project again with lognormal durations with the same expectations and SD 1. First compute the parameters $\mu$ and $\sigma$ of the underlying normal distributions, using the formulas for the lognormal expectation and variance in [Common Distributions](lecture12_variability-recap.ipynb#common-distributions).
# :::
#
# :::{exercise}
# :label: ex-5-3
#
# The budget of most companies is set without taking the variability of the numbers into account. Consider a simple budget:
#
# $$
# \text{profit} = \text{sales} \times (\text{price} - \text{variable costs}) - \text{fixed costs}.
# $$
#
# All components depend on market situations and are random; for simplicity we assume them to be independent and normally distributed, with mean and SD 10000 and 1000 for sales, 100 and 20 for the price, 80 and 10 for the variable costs, and 100000 and 20000 for the fixed costs.
#
# a. Simulate the expected profit and give a 95% CI.
#
# b. Could you have calculated the expected profit without simulation? Explain your answer.
#
# c. Estimate the probability of a loss, with a 95% CI.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 5, "Simulation."
