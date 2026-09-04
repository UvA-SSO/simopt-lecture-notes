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
# # Lecture 10: Modeling Tools and Solvers

# %% [markdown]
# So far we discussed solving LO and ILO problems, and the engines to solve them. However, we should realize that optimization specialists spend most of their time on modeling. Modeling is the translation of a real-life problem into a mathematical description that can be used to solve the problem. See the figure below for the steps in modeling. The time spent on modeling can be greatly reduced by using an appropriate modeling tool or language. As such, the existence of these modeling tools is considered to be of equal importance as the engines used to solve the models. To learn a modeling tool or language requires some time, but this easily pays off if you often build models for optimization problems.

# %% [markdown]
# ![Modeling steps](images/lecture10_fig6.10.png)

# %% [markdown]
# The best known algebraic modeling languages (AMLs) are AIMMS, AMPL, GAMS, LINDO, and MPL. Some of these languages are part of an integrated development environment (IDE) that simplifies the modeling even further. The problem entry in these AMLs is quite similar to mathematical notation (see the AMPL example in [Advanced Modeling](lecture9_advanced-modeling.ipynb)). Many of the tools and engines have free educational licenses, which makes it possible for students to learn and experiment. Even simpler to use is the NEOS server, a free cloud service which features a range of solvers to which one can submit optimization problems in a number of AML formats.
#
# The AMLs and associated IDEs allow experienced modelers to model and solve problems they encounter. However, engines can also be built into software dedicated to solve a particular class of problems such as navigation software. These types of problem-specific tools are called decision support systems (DSSs). They are geared towards a different class of users. While AMLs are used by experienced data scientists and OR consultants, DSSs are most often used by planners with domain knowledge, but with less or no background in optimization and modeling.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.6 "Modeling Tools."
