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
# description: "The simulation setting of Lectures 12 and 13, and the flaw of averages: why plugging in average inputs gives the wrong expected output."
# thumbnail: null
# ---
# # Lecture 12: Why Simulate?
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture12_why-simulation.ipynb)

# %% [markdown]
# In Lectures 8 to 11, every parameter of a model was known: durations, demands and distances were fixed numbers. In practice, many of them are uncertain. This notebook describes the setting of the simulation part of the course, where the inputs of a model are random, and shows with the flaw of averages why we cannot simply replace each random input by its average. The probability concepts it uses are summarized in [Variability (Recap)](lecture12_variability-recap.ipynb). The next notebooks cover how to sample random inputs ([Sampling a Random Variable](lecture12_sampling.ipynb)) and the two types of simulation ([Monte Carlo Simulation](lecture12_monte-carlo.ipynb) and [Discrete-Event Simulation](lecture12_discrete-event-simulation.ipynb)).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - describe the simulation setting: random inputs $X$, a model $r$ and the expected output $E[r(X)]$
# - explain the four steps of a simulation study
# - explain the flaw of averages, and recognize when $E[r(X)]$ differs from $r(EX)$

# %% [markdown]
# (simulation-setting)=
# ## The Simulation Setting
#
# The optimization problems so far had the form $\max r(\pi)$ subject to $\pi \in S$, where $S$ is the set of feasible decisions: a model $r$ computes the output of a decision $\pi$ for some fixed parameters. When some parameters are random, we collect them in a random variable $X$ (usually a vector, one component per random parameter), and the output $r(X, \pi)$ is random as well. A natural goal is then to maximize the expected output:
#
# $$
# \max_{\pi \in S} \ E[r(X, \pi)].
# $$
#
# In the problems of Lectures 12 and 13, $E[r(X, \pi)]$ can only be computed by **simulation**: imitating the random behavior of the real system many times on a computer. Lecture 13 is about choosing the decision $\pi$. In Lecture 12 the decision is fixed, so we drop $\pi$ and study the random output $r(X)$ itself: its expectation $E[r(X)]$, and probabilities such as $P(r(X) > 10)$. There are two types of simulation:
#
# - **Monte Carlo simulation**, when $r$ is a known function of a fixed number of random inputs, for example the finish time of a project as a function of the durations of its activities;
# - **discrete-event simulation**, for processes that are too complex to write down as one function, such as a queue of customers that evolves over time.
#
# Every simulation study goes through the same four steps:
#
# 1. **Fit a distribution** to historical data for each random input, for example a lognormal distribution for activity durations. This is part of the statistics part of the course; Canvas has Python code that finds the best-fitting distribution for a dataset. In these notes, the distributions are given.
# 2. **Sample** realizations $x_1, x_2, \dots, x_n$ of $X$ from the fitted distribution, so that they are in line with the real-life data ([Sampling a Random Variable](lecture12_sampling.ipynb)).
# 3. **Process** every sample in the model, giving the output samples $r(x_1), r(x_2), \dots, r(x_n)$.
# 4. **Analyze** the output samples statistically, for example by averaging them to estimate $E[r(X)]$ and computing a [confidence interval](lecture12_variability-recap.ipynb#confidence-intervals).
#
# Why fit a distribution in step 1, instead of reusing the historical data directly (called bootstrapping)? A fitted distribution gives an unlimited stream of new, realistic samples. Reusing the same limited history risks overfitting: once we optimize decisions (Lecture 13), the best decision for the historical data is not necessarily the best one for the future.

# %% [markdown]
# (flaw-of-averages)=
# ## The Flaw of Averages
#
# Why simulate at all? A tempting shortcut is to replace every random input by its expected value and to compute the output once, as if the inputs were fixed parameters. This shortcut gives the wrong answer, because in general
#
# $$
# E[r(X)] \ne r(EX).
# $$ (eq-flaw-of-averages)
#
# Assuming that they are equal is a mistake common enough to have its own name: the **flaw of averages** (Savage, 2012). Savage illustrates it with a drunk walking down the middle of a highway, swaying from side to side. The drunk's average position is on the middle line, where no car can hit them. But they are rarely at their average position, and one sway into a lane with a car is enough.
#
# Simulation avoids the flaw of averages without much effort: instead of computing $r$ once for the average input, it computes $r$ for many samples of $X$ and averages the outputs. That average estimates $E[r(X)]$ itself, not $r(EX)$.
#
# :::{note} Example: Project Planning
# :label: eg-flaw-project
#
# A project consists of three activities A, B and C. A and B can be done at the same time, and C can only start when both A and B are finished. The expected duration of each activity is 4.5 days. What is the expected finish time of the project?
#
# For durations $x_A$, $x_B$ and $x_C$, the project finishes at
#
# $$
# r(x_A, x_B, x_C) = \max(x_A, x_B) + x_C.
# $$
#
# Plugging in the expected durations gives $r(4.5, 4.5, 4.5) = 9$ days. But the durations are random, and the project waits for the slower of A and B. In a given realization, one of them is often late, and that delay always passes on to the finish time, while an activity that is early does not help if the other one is late. So the expected finish time $E[\max(X_A, X_B) + X_C]$ is more than 9 days. How much more depends on the distributions; [Monte Carlo Simulation](lecture12_monte-carlo.ipynb#project-planning-example) estimates it for lognormal durations.
#
# With more activities that run at the same time, it becomes ever more likely that at least one of them is unusually long, so the gap grows. A project whose last activity waits for a thousand others will almost surely be delayed by one of them.
# :::
#
# Equality does hold when $r$ is linear: $E[aX + bY + c] = aEX + bEY + c$, by the [rules for sums](lecture12_variability-recap.ipynb#sums-of-rvs). A maximum, a minimum (sales are the minimum of demand and stock), a threshold or a product of random inputs breaks it. For such a model there is usually no formula for $E[r(X)]$, especially when $X$ has many components, and simulation is the way to compute it.
#
# :::{exercise}
# :label: ex-flaw-of-averages
#
# Let $X$ take the values 0, 1 and 2, each with probability $1/3$.
#
# a. Compute $E[X^2]$ and $(EX)^2$. Which one is larger?
#
# b. A shop has 1 item in stock, and the demand is $X$. Compute the expected sales $E[\min(X, 1)]$ and compare it with $\min(EX, 1)$.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 5, "Simulation."
# - Savage, S. (2012). *The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty*. Wiley.
