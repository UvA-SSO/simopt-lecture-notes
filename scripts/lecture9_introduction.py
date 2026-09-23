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
# # Lecture 9: Applications of (Integer) Linear Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_introduction.ipynb)

# %% [markdown]
# This lecture broadens the range of real-life problems you can cast as (integer) linear
# optimization, and introduces the modeling tricks that make that possible, partly for
# their own sake and partly as inspiration for modeling your own problems.
# [Why Linearity Matters, and What Integrality Buys
# Back](lecture8_integer-optimization.ipynb#why-linearity-matters), in
# [Integer Optimization](lecture8_integer-optimization.ipynb), explains why we lean so
# heavily on these models in the first place; this lecture puts that into practice.
#
# **Learning outcomes**
#
# On completion of this lecture, you will be able to:
#
# - recognize and formulate the transportation and transshipment problems;
# - formulate set cover, set covering, and shift-scheduling problems as ILO models;
# - model fixed costs and either/or conditions with big-M constraints and binary indicator
#   variables, and formulate a single-machine scheduling problem as an ILO model.

# %% [markdown]
# ## Applications in This Lecture
#
# - [Transportation and Transshipment](lecture9_transportation.ipynb): shipping goods
#   between sources and destinations, or through intermediate nodes, at minimum cost.
# - [Set Covering and Shift Scheduling](lecture9_covering.ipynb): choosing the cheapest
#   set of shifts (or facilities) that between them cover every point that needs covering.
# - [Machine Scheduling](lecture9_machine-scheduling.ipynb): the modeling tricks (big-M,
#   disjunctive constraints) that build up to scheduling jobs on a single machine.
#
# [Exercises](lecture9_exercises.ipynb) collects the larger, independent exercises for all
# three.
