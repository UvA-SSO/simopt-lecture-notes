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
# # Lecture 8: Introduction to Business Analytics

# %% [markdown]
# This part of the course is about **optimization** and **simulation**: turning data and
# predictions into good decisions. This first notebook places that in context — what kind
# of analytics it is, why it is valuable, what we will cover, and how a real-life problem
# becomes a mathematical model. The technical material starts in
# [Linear Optimization](lecture8_linear-optimization.ipynb).

# %% [markdown]
# ## Prescriptive Analytics and Optimization
#
# Business analytics is usually split into a sequence of activities, each answering a
# different question about a business process:
#
# - **descriptive** analytics — *what happened?* (reporting, visualization, summary
#   statistics);
# - **diagnostic** analytics — *why did it happen?*;
# - **predictive** analytics — *what is likely to happen?* (forecasting, machine
#   learning);
# - **prescriptive** analytics — *what should we do about it?*
#
# This course is about the last step. Prescriptive analytics determines which **decision**
# leads to the best outcome, given the data and the predictions produced by the earlier
# steps. Its main tool is **mathematical optimization**, and that is what "the simulation
# and optimization part" of this course is about.
#
# :::{note} Example: Hotel Revenue Management
# :label: eg-1-1
#
# A hotel chain analyzes its reservations to find patterns: which are the busiest days of
# the week, what is the impact of events in the city, is there a seasonal pattern
# (*descriptive*). The patterns feed a forecast of demand per room-price class
# (*predictive*). That forecast is the input to an algorithm that sets room prices each day
# so as to maximize expected revenue (*prescriptive*).
# :::
#
# Typical questions that prescriptive analytics answers:
#
# - How should we assign employees to shifts to minimize labour cost while covering demand?
# - How should delivery trucks be routed to minimize total distance?
# - How much stock should we order, and when?
# - Which portfolio of investments best trades off return against risk?
# - How do we split school children into groups with the best group cohesion?
#
# Prescriptive analytics has a large potential business value, but it is harder and less
# widely adopted than descriptive or predictive analytics. For an organization that does
# adopt it, that gap is exactly where a competitive advantage can be found.
#
# :::{note} Example: Debt Collection
# :label: eg-1-3
#
# A debt-collection agency wants to use its calls to debtors more effectively. It enriches
# its payment data with external data on household composition and neighbourhood
# characteristics (*descriptive*), fits a model that predicts, per debtor, the probability
# of paying off the debt given the actions taken (*predictive*), and then chooses the best
# action for each debtor (*prescriptive*).
# :::

# %% [markdown]
# ## What This Course Covers
#
# The goals of this part of the course are to:
#
# - introduce the **process of optimization**: how a business problem becomes a model that
#   can be solved;
# - cover optimization techniques for **deterministic** problems (no uncertainty) and for
#   **stochastic** problems (with uncertainty), and the role of **simulation** in
#   evaluating stochastic models;
# - introduce optimization and simulation **software**;
# - make you able to model, solve, and **interpret the results** of relatively simple
#   problems, and to recognize optimization opportunities in practice.
#
# The lectures are:
#
# | Lecture | Topic | |
# |---|---|---|
# | 8  | Linear optimization | *deterministic models* |
# | 9  | Applications and advanced modeling | |
# | 10 | Modeling tools and solvers | |
# | 11 | Algorithms and complexity | |
# | 12 | Simulation | *stochastic models* |
# | 13 | Simulation optimization | |
#
# These notes are the deeper, self-paced companion to the lectures: the lecture moves
# quickly over the details, and you can work through the reasoning here at your own pace,
# before or after class.

# %% [markdown]
# ## From Real-Life Problem to Model
#
# Every optimization problem in this course has the same three ingredients. A real-life
# problem consists of:
#
# - a **decision** we get to make — which becomes the **decision variables**;
# - a **system** that constrains what decisions are allowed — which becomes the
#   **constraints**;
# - an **outcome** we care about — which becomes the **objective function**.
#
# The real-life question, *"which feasible decision gives the best outcome?"*, then becomes
# a precise mathematical one. The value of this translation is that all kinds of problems,
# from very different domains, can be expressed in the same "universal modeling language"
# of mathematics, so that the same general solution methods and software apply to all of
# them.
#
# The workflow is a loop rather than a straight line:
#
# > **real-life problem** → *(modeling)* → **model** → *(solving)* → **decision**
#
# with **data** and **data analysis** feeding the model's parameters, and with results
# often sending you back to refine the model. Most of the effort — and most of this course
# — is in the modeling step, not the solving step.
#
# > "The formulation of a problem is often more essential than its solution, which may be
# > merely a matter of mathematical or experimental skill."
# > — Albert Einstein & Leopold Infeld, *The Evolution of Physics* (1938)
#
# In [Linear Optimization](lecture8_linear-optimization.ipynb) we make this concrete with a
# four-step modeling approach: study the problem in detail, define the decision variables,
# define the objective, define the constraints.

# %% [markdown]
# ## Tooling
#
# This course uses **Python**. Python's built-in functions are deliberately minimal; the
# data-analysis functionality comes from libraries, most importantly `numpy` (arrays and
# numerical computing), `pandas` (tabular data), `matplotlib` (plotting), and `scipy`
# (statistics and scientific computing). For optimization we will add `pulp`
# (see [Modeling Tools and Solvers](lecture10_modeling-tools.ipynb)). These notes are
# themselves Jupyter notebooks, combining code, its output, and explanation.
#
# For the actual optimization, Python calls a separate **solver** engine. The best-known
# solvers, Gurobi and CPLEX, are proprietary; CBC is a widely used open-source one, and it
# is the default that `pulp` uses.

# %% [markdown]
# ## Further Reading
#
# This notebook only sets the scene. Koole (2019), Chapter 1, goes further into the
# non-technical background of business analytics: the historical roots in statistics,
# artificial intelligence, and operations research; the difference between business
# analytics, data science, and "big data"; structured versus unstructured data; supervised
# versus unsupervised learning; the "T-shaped" and "Π-shaped" analyst; organizational
# analytics-maturity models; and the legal and ethical aspects of working with data. None
# of that is needed for the optimization and simulation material that follows, but it is
# worth reading once.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 1, "Introduction."
# - Davenport, T.H., & Harris, J.G. (2007). *Competing on Analytics: The New Science of
#   Winning*. Harvard Business School.
# - Einstein, A., & Infeld, L. (1938). *The Evolution of Physics*. Cambridge University
#   Press.
# - Powell, S.G., Baker, K.R., & Lawson, B. (2009). "Impact of errors in operational
#   spreadsheets." *Decision Support Systems*, 46:126–132.
