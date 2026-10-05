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
# description: "Sampling a random variable: the inverse transform method, and sampling in practice with a numpy random generator and a seed."
# thumbnail: null
# ---
# # Lecture 12: Sampling a Random Variable
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture12_sampling.ipynb)

# %% [markdown]
# Step 2 of a [simulation study](lecture12_why-simulation.ipynb) is to sample realizations of the random inputs from their distributions. This notebook first explains the method behind almost all sampling, the inverse transform method, which you can also apply by hand. It then shows how to sample in Python with `numpy`, and how a seed makes the results reproducible. [Monte Carlo Simulation](lecture12_monte-carlo.ipynb) uses these samples.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - explain the inverse transform method and why it works
# - sample by hand from a continuous or discrete distribution, given uniform[0, 1] numbers
# - sample from common distributions with a `numpy` random generator, and use a seed to make results reproducible

# %% [markdown]
# ## The Inverse Transform Method
#
# Every programming language can generate numbers $u$ that are uniformly distributed on $[0, 1]$. The **inverse transform method** (ITM) turns them into samples from any distribution with cdf $F$: sample $u$ and return the $x$ with $F(x) = u$. We write this $x$ as $F^{-1}(u)$. [](#fig-itm) shows the idea: go from $u$ on the vertical axis horizontally to the cdf, and then down to the $x$-axis.

# %% tags=["remove-cell"] label="itm-exponential"
import numpy as np
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
AXIS_STYLE = {
    "gridcolor": "rgba(128,128,128,0.25)",
    "zerolinecolor": "rgba(128,128,128,0.5)",
}
# static pictures: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}

x_grid = np.linspace(0, 10, 300)
itm_fig = go.Figure()
itm_fig.add_trace(
    go.Scatter(
        x=x_grid,
        y=1 - np.exp(-0.5 * x_grid),
        mode="lines",
        line={"color": "#1f77b4", "width": 3},
        name="cdf F(x) = 1 − e^(−x/2)",
    )
)
for u_value, color in [(0.3, "#ff7f0e"), (0.9, "#d62728")]:
    x_value = -np.log(1 - u_value) / 0.5
    itm_fig.add_trace(
        go.Scatter(
            x=[0, x_value, x_value],
            y=[u_value, u_value, 0],
            mode="lines+markers",
            line={"color": color, "dash": "dash", "width": 2},
            marker={"size": [9, 0, 9], "color": color},
            name=f"u = {u_value} gives x = {x_value:.2f}",
        )
    )
itm_fig.update_layout(**PLOT_LAYOUT)
itm_fig.update_xaxes(title="x", range=[0, 10], **AXIS_STYLE)
itm_fig.update_yaxes(title="u = F(x)", range=[0, 1.02], **AXIS_STYLE)
itm_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #itm-exponential
# :label: fig-itm
#
# The inverse transform method for the exponential distribution with rate 0.5: the uniform numbers $u = 0.3$ and $u = 0.9$ become the samples $F^{-1}(0.3) = 0.71$ and $F^{-1}(0.9) = 4.61$.
# :::
#
# **Why it works.** Let $U$ be uniform on $[0, 1]$, so $P(U \le u) = u$. Because $F$ is non-decreasing, $F^{-1}(U) \le x$ holds exactly when $U \le F(x)$. So for every $x$,
#
# $$
# P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x) = P(X \le x).
# $$
#
# The sample $F^{-1}(U)$ has the same cdf as $X$, so it has the same distribution.
#
# **Intuition.** Uniform numbers are spread evenly over the vertical axis. Where $F$ is steep, a large part of the vertical axis maps to a short interval of $x$-values, so many samples land there. $F$ is steep exactly where the density $f$ is high, so the samples follow the shape of the pdf. In [](#fig-itm), most $u$-values end up at small $x$, where the exponential density is highest.
#
# **Example: uniform[$a$, $b$].** The cdf is $F(x) = (x - a)/(b - a)$ on $[a, b]$. Solving $F(x) = u$ gives
#
# $$
# F^{-1}(u) = a + (b - a)u.
# $$
#
# **Example: exponential.** A component's lifetime $X$ (in years) is exponential with rate $\lambda = 0.5$, so mean $1/\lambda = 2$ years, and $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$. Solving $F(x) = u$:
#
# $$
# 1 - e^{-\lambda x} = u \quad\Longrightarrow\quad x = -\frac{\ln(1 - u)}{\lambda} = F^{-1}(u).
# $$
#
# Since $1 - U$ is uniform on $[0, 1]$ as well, $-\ln(u)/\lambda$ also gives exponential samples, and is often used instead.
#
# **Example: a discrete distribution.** A delivery arrives 1, 2 or 3 days late, with probabilities 0.2, 0.5 and 0.3. The cdf is a step function with $F(1) = 0.2$, $F(2) = 0.7$ and $F(3) = 1$. A value $u$ usually lies between two steps, so we take the smallest $x$ with $F(x) \ge u$:
#
# $$
# F^{-1}(u) = \min\{x : F(x) \ge u\} =
# \begin{cases}
# 1 & \text{if } 0 \le u \le 0.2, \\
# 2 & \text{if } 0.2 < u \le 0.7, \\
# 3 & \text{if } 0.7 < u \le 1.
# \end{cases}
# $$
#
# In words: split $[0, 1]$ into consecutive intervals whose lengths are the probabilities of the outcomes, sample $u$, and return the outcome whose interval contains $u$. For example, $u = 0.45$ gives a delay of 2 days.
#
# :::{exercise}
# :label: ex-itm-by-hand
#
# Use the uniform numbers $u = 0.15$ and $u = 0.75$.
#
# a. Sample two lifetimes from the exponential distribution with mean 2 years.
#
# b. Sample two delivery delays from the discrete distribution above.
#
# c. A random variable on $[0, 1]$ has cdf $F(x) = x^2$. Give $F^{-1}(u)$ and sample two values.
# :::

# %% [markdown]
# ## Sampling in Practice with numpy
#
# ### Random Number Generators and Seeds
#
# A computer generates uniform numbers with a **pseudo-random number generator** (PRNG): a formula that produces a sequence of numbers that looks random and passes statistical tests for randomness, but is completely determined by its starting value, the **seed**. True randomness exists too, for example based on atmospheric noise, but it is slower and rarely needed.
#
# That a seed determines all numbers is useful. A simulation with a fixed seed gives the same results every time it runs, so you can reproduce and check results, find errors, and compare two decisions under exactly the same randomness (Lecture 13 makes use of this).
#
# In `numpy`, you create a random generator with `np.random.default_rng(seed)` and draw all random numbers from it. Its method `uniform(low, high, size)` gives `size` uniform samples:

# %%
import numpy as np

rng = np.random.default_rng(seed=42)
print(rng.uniform(0, 1, 3))
print(rng.uniform(0, 1, 3))

# %% [markdown]
# The second call continues the sequence, so it gives new numbers. A new generator with the same seed starts the same sequence again:

# %%
rng = np.random.default_rng(seed=42)
print(rng.uniform(0, 1, 3))

# %% [markdown]
# A different seed gives a different sequence. Without a seed, `np.random.default_rng()` takes a seed from the operating system, so every run of the notebook gives different numbers.
#
# :::{note} Older numpy Code
# The slides and much code online use the older functions `np.random.seed(1)`, `np.random.rand()` (a uniform[0, 1] sample) and `np.random.randn()` (a standard normal sample). They still work, but `numpy` recommends a generator from `np.random.default_rng`, as in these notes.
# :::
#
# ### Sampling with the Inverse Transform Method
#
# The ITM formulas work on whole arrays of uniform numbers at once. Here are 10,000 exponential lifetimes with rate 0.5:

# %%
rate = 0.5
u = rng.uniform(0, 1, 10000)
lifetimes = -np.log(1 - u) / rate
print(lifetimes[:5])
print("sample mean:", lifetimes.mean())

# %% [markdown]
# The sample mean is close to the expectation $1/\lambda = 2$, and the histogram in [](#fig-itm-histogram) has the shape of the exponential density.

# %% tags=["remove-cell"] label="itm-histogram"
hist_fig = go.Figure()
hist_fig.add_trace(
    go.Histogram(
        x=lifetimes,
        histnorm="probability density",
        xbins={"start": 0, "end": 16, "size": 0.5},
        marker={"color": "rgba(31,119,180,0.6)", "line": {"width": 0}},
        name="histogram of the samples",
    )
)
pdf_grid = np.linspace(0, 16, 200)
hist_fig.add_trace(
    go.Scatter(
        x=pdf_grid,
        y=rate * np.exp(-rate * pdf_grid),
        mode="lines",
        line={"color": "#d62728", "width": 3},
        name="exponential pdf",
    )
)
hist_fig.update_layout(**PLOT_LAYOUT, bargap=0.05)
hist_fig.update_xaxes(title="lifetime (years)", range=[0, 16], **AXIS_STYLE)
hist_fig.update_yaxes(title="density", **AXIS_STYLE)
hist_fig.show(config=PLOT_CONFIG)

# %% [markdown]
# :::{figure} #itm-histogram
# :label: fig-itm-histogram
#
# Histogram of 10,000 exponential samples from the inverse transform method, with the exponential pdf (rate 0.5).
# :::
#
# For the delivery delays, `np.where(condition, a, b)` takes, for every sample, `a` where the condition holds and `b` elsewhere. Nesting it follows the three intervals of the ITM:

# %%
u = rng.uniform(0, 1, 10000)
delays = np.where(u <= 0.2, 1, np.where(u <= 0.7, 2, 3))
print(delays[:10])
for days in [1, 2, 3]:
    print(f"fraction of {days}-day delays:", (delays == days).mean())

# %% [markdown]
# `delays == days` gives an array of `True`/`False` values, and its mean is the fraction of `True` (which counts as 1). The fractions are close to 0.2, 0.5 and 0.3.
#
# (sampling-directly)=
# ### Sampling Directly
#
# For common distributions, the generator has a method that samples directly (internally, often with the ITM or a similar transformation):
#
# | distribution | `numpy` code for `n` samples | note |
# |---|---|---|
# | uniform[$a$, $b$] | `rng.uniform(a, b, n)` | |
# | normal | `rng.normal(mean, sd, n)` | the SD, not the variance |
# | exponential | `rng.exponential(mean, n)` | the mean $1/\lambda$, not the rate $\lambda$ |
# | lognormal | `rng.lognormal(mu, sigma, n)` | `mu` and `sigma` of the underlying normal distribution |
# | Poisson | `rng.poisson(mean, n)` | |
# | binomial | `rng.binomial(trials, p, n)` | |
# | integers $a, \dots, b - 1$, equally likely | `rng.integers(a, b, n)` | `b` itself is excluded: a die roll is `rng.integers(1, 7)` |
# | a discrete distribution | `rng.choice([1, 2, 3], n, p=[0.2, 0.5, 0.3])` | values and their probabilities |
#
# Without `n`, each method returns a single number instead of an array. The direct versions of the two examples above:

# %%
lifetimes = rng.exponential(2, 10000)
delays = rng.choice([1, 2, 3], 10000, p=[0.2, 0.5, 0.3])
print("mean lifetime:", lifetimes.mean())
print("mean delay:", delays.mean())

# %% [markdown]
# :::{exercise}
# :label: ex-numpy-sampling
#
# a. Write one line of code that samples 1000 rolls of a die.
#
# b. Activity durations are lognormal with parameters $\mu = 1$ and $\sigma = 1$. What is the expected duration? Sample 100,000 durations with `rng.lognormal` and check your answer with the sample mean.
#
# c. Run the cell that creates `rng = np.random.default_rng(seed=42)` and the cell after it again. Do you get the same numbers? What happens if you remove `seed=42`?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 5, "Simulation."
# - Ross, S.M. (1996). *Simulation*, 6th ed. Academic Press.
