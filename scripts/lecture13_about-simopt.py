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
# description: "The simulation optimization setting, why it is harder than deterministic optimization, and the four types of problems discussed in Lecture 13."
# thumbnail: null
# ---
# # Lecture 13: About Simulation Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_about-simopt.ipynb)

# %% [markdown]
# Lectures 8 to 11 optimized models without randomness, and Lecture 12 simulated models with randomness for one fixed decision. Lecture 13 combines the two: we choose a decision whose performance is random and can only be evaluated by simulation. This notebook describes that setting, explains why it is harder than deterministic optimization, and gives an overview of four types of problems. The next notebooks discuss a strategy for each type: [Comparing Scenarios](lecture13_comparing-scenarios.ipynb), [Ranking and Selection](lecture13_ranking-and-selection.ipynb), [Local Search](lecture13_local-search.ipynb) and [Gradient Methods](lecture13_gradient-methods.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - describe the simulation optimization setting and give examples of it
# - explain why noise makes simulation optimization harder than deterministic optimization
# - recognize which of the four types of simulation optimization problems a problem belongs to

# %% [markdown]
# (simopt-setting)=
# ## The Setting
#
# A real-life problem has random parameters $X$ (demands, travel times, arrival moments, ...) and a decision $\pi$ that we choose from a set of options $S$. As in [Why Simulate?](lecture12_why-simulation.ipynb#simulation-setting), a model $r$ gives the random output $r(X, \pi)$, and the goal is to maximize its expectation:
#
# $$
# \max_{\pi \in S} \ E[r(X, \pi)].
# $$
#
# The key feature is that $E[r(X, \pi)]$ can only be evaluated by simulating $r(X, \pi)$: there is no formula that we could optimize directly. This setting is called **simulation optimization**, also known as optimization by simulation, simulation-based optimization, or simply simopt. Replacing $X$ by its expectation and solving $\max_{\pi \in S} r(EX, \pi)$ instead is not a way out: by the [flaw of averages](lecture12_why-simulation.ipynb#flaw-of-averages), $E[r(X, \pi)]$ and $r(EX, \pi)$ can be very different, and so can the decisions that maximize them.
#
# Simulation optimization lets us evaluate decisions in a realistic model of a process, with all its randomness, and that makes it useful in practice: should a bank open an extra desk, which layout of a factory gives the highest throughput, how long should a traffic light stay red?
#
# :::{note} Example: Extending Earlier Simulation Examples
# :label: eg-8-1
#
# The examples from Lecture 12 can be extended with decisions: the [budget exercise](lecture12_monte-carlo.ipynb#ex-5-3) can compare two strategic decisions of a company, and in the [queue](lecture12_discrete-event-simulation.ipynb#example-a-queue) we can choose the number of servers.
# :::
#
# :::{note} Example: The Newsvendor Problem
# :label: eg-8-2
#
# A newsvendor buys newspapers every morning and sells them during the day. Newspapers that are left at the end of the day are worthless. Demand is random, so ordering too many papers costs money on papers that are thrown away, and ordering too few loses sales. How many newspapers should the newsvendor buy to maximize the expected profit? This problem returns in every notebook of this lecture.
# :::
#
# :::{note} Random Constraints
# We could generalize the problem further, from $\max_{\pi \in S} E[r(X, \pi)]$ to a problem with random constraints $E[g_j(X, \pi)] \le b_j$, for example that the probability that a customer waits longer than 15 minutes is at most 5%. Such problems are very hard to solve, and in this lecture the only constraint is $\pi \in S$.
# :::

# %% [markdown]
# (simopt-challenges)=
# ## Why Simulation Optimization Is Hard
#
# Simulation optimization is harder than the deterministic optimization of Lectures 8 to 11, for two reasons that have nothing to do with the model itself:
#
# - There are no optimality guarantees. We only see estimates of $E[r(X, \pi)]$, so we can never be sure that one solution is better than another.
# - Each evaluation can be slow, especially for a discrete-event simulation of a complex process.
#
# Because of the second reason, we usually have a limited **simulation budget** $m$: the total number of simulation runs we can do, say 10,000. The central question of this lecture is how to spend this budget well. Compare it with testing slot machines in a casino with 100 coins: how do you spend the coins to find the machine with the highest expected profit? Playing every machine equally often wastes coins on machines that are clearly bad, while playing only the machine that looked best after one try may miss a better one.
#
# The noise in the estimates causes two problems. Because of the noise we might not recognize the signal and pick the wrong solution. And because of the noise we might overestimate the value: instead of selecting the solution with the highest value we pick the one with the highest random component, so the estimated value of the chosen solution is typically too high. To avoid this we could simulate each solution many more times, but this might take too much time. As a result, we cannot avoid these problems completely, but we can spend the budget better than by splitting it equally over all solutions. [Ranking and Selection](lecture13_ranking-and-selection.ipynb#option-1) shows both problems for a newsvendor.

# %% [markdown]
# (four-types)=
# ## Four Types of Problems
#
# How we spend the simulation budget depends on the size and shape of $S$. We distinguish four types of problems, each with its own strategy:
#
# | Shape of $S$ | Strategy |
# |---|---|
# | $\lvert S \rvert = 2$ | [comparing scenarios](lecture13_comparing-scenarios.ipynb) |
# | $S$ discrete and small | [ranking and selection](lecture13_ranking-and-selection.ipynb) |
# | $S$ discrete but large, possibly infinite (e.g., a grid) | [local search](lecture13_local-search.ipynb) |
# | $S$ continuous (e.g., an interval) | [gradient methods](lecture13_gradient-methods.ipynb) |
#
# The next notebooks discuss these strategies in this order, from $|S| = 2$ to continuous $S$. For each type we give one simple method, to show the ideas; the scientific literature contains many more advanced methods. The techniques of these notebooks are also the building blocks for more challenging problems, for example problems with both discrete and continuous decisions, such as choosing the number of servers in a queue together with the length of their shifts.
#
# :::{exercise}
# :label: ex-four-types
#
# To which of the four types does each problem belong?
#
# a. A factory chooses between two layouts of its production line.
#
# b. A traffic light at a large crossroads stays red for $\pi$ seconds, where $\pi$ can be any number between 20 and 90.
#
# c. A call center chooses how many agents to schedule in each of the 48 half-hour intervals of a day.
#
# d. The newsvendor of [](#eg-8-2) has space for at most 20 newspapers.
# :::

# %% [markdown]
# ## Additional Reading
#
# There are few accessible books on simopt. Nelson (2013) is a textbook on simulation that includes a chapter on simopt; Fu (2002) is an accessible introduction to the subject.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 8, "Simulation Optimization," introduction.
# - Fu, M.C. (2002). "Optimization for simulation: Theory vs. practice." *INFORMS Journal on Computing*, 14:192–215.
# - Nelson, B.L. (2013). *Foundations and Methods of Stochastic Simulation*. Springer.
