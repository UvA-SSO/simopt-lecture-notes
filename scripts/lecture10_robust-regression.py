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
# description: "Robust regression as an LO model: fitting a line by minimizing the sum of absolute errors, and the trick that makes absolute values linear."
# thumbnail: null
# ---
# # Lecture 10: Robust Regression
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_robust-regression.ipynb)

# %% [markdown]
# This notebook covers robust regression: fitting a line through data points such that
# the sum of the absolute prediction errors is as small as possible, which makes the line
# far less sensitive to outliers than ordinary least squares. The absolute value is not
# linear, so the model needs a trick that splits each error into a positive and a
# negative part; the same trick linearizes absolute values in other models. Like the
# applications of [Lecture 9](lecture9_introduction.ipynb) and
# [Multi-Period Inventory Planning](lecture10_multi-period.ipynb), the problem is
# introduced in the same steps: a practical motivation, the generic model built up with
# the [four modeling steps](lecture8_linear-optimization.ipynb#modeling-approach), the
# model for a concrete example, its solution in pulp, and possible extensions.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate a robust regression problem as an LO model;
# - linearize absolute values in a model with the $e = e^+ - e^-$ split;
# - adapt the model to quantile regression.

# %% [markdown]
# (robust-regression)=
# ## Robust Regression
#
# ### Practical Motivation
#
# A factory wants to predict how long a production order will take from its size, to
# promise delivery dates. It fits a line through the data of past orders. A few of those
# orders took far longer than usual, for example because a machine broke down.
# Ordinary least-squares (OLS) regression minimizes the sum of *squared* prediction
# errors, so such outliers have a large influence and pull the line toward them, just as
# a single extreme value pulls the mean. Minimizing the sum of *absolute* errors instead
# gives a line that is far less sensitive to outliers, the line analogue of the median.
# This is called **robust regression** (or least absolute deviation regression).

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# Given $n$ data points $(x_i, y_i)$, $i = 1, \dots, n$, find the line $a + bx$ that
# minimizes the sum of the absolute prediction errors $|y_i - (a + b x_i)|$.
#
# As an example, take the five data points
#
# | $i$ | 1 | 2 | 3 | 4 | 5 |
# |---|---|---|---|---|---|
# | $x_i$ | 2 | 4 | 6 | 8 | 10 |
# | $y_i$ | 5 | 4 | 9 | 3 | 7 |
#
# #### Decision Variables
#
# We choose the line, so its intercept $a$ and slope $b$ are decision variables. Both
# may be negative, so unlike in most models so far they have no sign constraint. It
# helps to also introduce the prediction error of each data point as a variable,
#
# $$
# e_i = y_i - (a + b x_i), \quad i = 1, \dots, n.
# $$
#
# #### Objective
#
# We want to minimize $\sum_{i=1}^{n} |e_i|$. The absolute value is not linear, so this
# is not yet an LO model. There is a standard trick to fix this. Write each error as the
# difference of two non-negative variables,
#
# $$
# e_i = e_i^+ - e_i^-, \quad e_i^+, e_i^- \ge 0,
# $$
#
# and replace $|e_i|$ by $e_i^+ + e_i^-$ in the objective, which is linear:
#
# $$
# \min \sum_{i=1}^{n} (e_i^+ + e_i^-).
# $$
#
# Why is this allowed? The same $e_i$ can be written as $e_i^+ - e_i^-$ in many ways,
# for example $2 = 2 - 0 = 5 - 3$. But if both $e_i^+$ and $e_i^-$ were positive, we
# could lower both by the same amount: $e_i$ stays the same and the objective goes down.
# So at the optimum, at least one of the two is 0, and then $e_i^+ + e_i^- = |e_i|$. If
# $e_i^+ > 0$, the data point lies $e_i^+$ above the line; if $e_i^- > 0$, it lies
# $e_i^-$ below the line.
#
# #### Constraints
#
# The only constraints connect the errors to the line:
#
# $$
# e_i^+ - e_i^- = y_i - (a + b x_i), \quad i = 1, \dots, n,
# $$
#
# together with $e_i^+, e_i^- \ge 0$. The variables $a$ and $b$ are free.
#
# #### Complete LO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} (e_i^+ + e_i^-) \\
# \text{s.t.} \quad & e_i^+ - e_i^- = y_i - (a + b x_i), \quad i = 1, \dots, n \\
# & e_i^+, e_i^- \ge 0, \quad i = 1, \dots, n.
# \end{aligned}
# $$

# %% [markdown]
# ### Modeling the Example
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{5} (e_i^+ + e_i^-) \\
# \text{s.t.} \quad & e_1^+ - e_1^- = 5 - (a + 2b) \\
# & e_2^+ - e_2^- = 4 - (a + 4b) \\
# & e_3^+ - e_3^- = 9 - (a + 6b) \\
# & e_4^+ - e_4^- = 3 - (a + 8b) \\
# & e_5^+ - e_5^- = 7 - (a + 10b) \\
# & e_i^+, e_i^- \ge 0, \quad i = 1, \dots, 5.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# A variable without `lowBound` is free in pulp (its bounds default to $-\infty$ and
# $\infty$), which is what we need for the intercept and the slope. The data points are
# stored as numpy arrays, which pulp's expressions accept like ordinary numbers.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pulp

xs = np.array([2, 4, 6, 8, 10])
ys = np.array([5, 4, 9, 3, 7])
points = range(len(xs))

robust_regression = pulp.LpProblem(
    name="robust_regression", sense=pulp.LpMinimize
)
intercept = pulp.LpVariable(name="a")
slope = pulp.LpVariable(name="b")
e_pos = [pulp.LpVariable(name=f"ep_{k + 1}", lowBound=0) for k in points]
e_neg = [pulp.LpVariable(name=f"en_{k + 1}", lowBound=0) for k in points]

robust_regression += pulp.lpSum(e_pos[k] + e_neg[k] for k in points)
for k in points:
    prediction = intercept + slope * xs[k]
    robust_regression += ys[k] - prediction == e_pos[k] - e_neg[k]

robust_regression.solve(pulp.PULP_CBC_CMD(msg=False))
a_hat, b_hat = intercept.value(), slope.value()
print("status:", pulp.LpStatus[robust_regression.status])
print(f"line: y = {a_hat:.2f} + {b_hat:.2f} x")
print("sum of absolute errors:", robust_regression.objective.value())

# %% tags=["remove-cell"] label="robust-fit"
plt.figure(figsize=(5, 3.5))
plt.scatter(xs, ys)
grid = np.linspace(xs.min(), xs.max(), 50)
plt.plot(grid, a_hat + b_hat * grid, color="grey")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# :::{figure} #robust-fit
# :label: fig-robust-fit
#
# The five data points of the example and the line $y = 4.5 + 0.25x$ that minimizes the
# sum of absolute errors.
# :::
#
# The optimal line passes exactly through the first and the last data point. That is no
# coincidence: as in every LO problem, there is an optimal solution in a corner point,
# and for this model that means a line through (at least) two of the data points.

# %% [markdown]
# ### Extensions
#
# #### Quantile Regression
#
# Weighting the two parts of the error differently,
# $\min\, p \sum_i e_i^+ + (1 - p) \sum_i e_i^-$ with $0 < p < 1$, tilts the line toward
# the upper or lower points: this is quantile regression ([](#fig-quantile-regression)).
# With $p = 0.5$ it is the robust regression above; with $p = 0.9$, about 90% of the data
# points lie below the line, which is useful for promising delivery times that are met
# in 90% of the cases.
#
# :::{figure} images/lecture9_fig6.11.png
# :label: fig-quantile-regression
#
# Quantile regression: the fitted line for different quantile levels $p$.
# :::
#
# #### Absolute Values in Other Models
#
# The same $x = x^+ - x^-$ split linearizes any $|x|$ that appears in an objective that
# is minimized with a non-negative coefficient (or maximized with a non-positive one).
# The trick also works when the absolute value is a *penalty* rather than the whole
# objective. For the [product-mix problem](lecture8_linear-optimization.ipynb), suppose we dislike making the two products in
# very different quantities and add $-|x - y|$ to the profit. Introduce
# $\delta^+, \delta^- \ge 0$ with $x - y = \delta^+ - \delta^-$ and subtract
# $\delta^+ + \delta^-$ from the objective.
#
# :::{exercise}
# :label: ex-6-16
#
# Numbers $a_1, \dots, a_n$ are given; find $x$ minimizing $\sum_i |x - a_i|$. Formulate as
# an LO and solve in pulp for 1, 2, 3, 5, 8, 10, 20, 35, 100. How do you interpret the
# result?
# :::
#
# :::{exercise}
# :label: ex-6-17
#
# Take [the shift-scheduling example](lecture9_covering.ipynb#shift-scheduling).
# Instead of requiring the staffing level to be met in every interval, minimize the sum of
# absolute differences between staffing and requirement. Formulate as an ILO and solve
# with pulp.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.7 "Modeling Tricks"
#   (quantile regression). Spreadsheet material replaced with pulp.
# - PuLP documentation: https://coin-or.github.io/pulp/
