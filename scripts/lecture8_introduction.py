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
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_introduction.ipynb)

# %% [markdown]
# This part of the course is about optimization and simulation: turning data into good
# decisions with business analytics (BA). This first notebook places that in context: what
# kind of analytics it is, how it relates to data science, and how a real-life problem
# becomes a mathematical model.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - relate business analytics to data science and big data, and to its roots in
#   statistics, artificial intelligence, and optimization;
# - place prescriptive analytics among the descriptive/predictive/prescriptive
#   sequence, and explain what distinguishes it;
# - describe the steps of a BA project, from data collection to implementation, and which
#   field of study each step draws on;
# - translate a real-life decision problem into the decision/objective/constraints
#   ingredients of a mathematical model.

# %% [markdown]
# ## What Is Business Analytics?
#
# According to Wikipedia, "business analytics refers to the skills, technologies, and
# practices for continuous iterative exploration and investigation of past business
# performance to gain insight and drive business planning." In short, **business analytics
# (BA)** is a rational, fact-based approach to decision making. These facts come from data,
# therefore BA is about the science and the skills to turn data into decisions. The science
# is mostly *statistics*, *artificial intelligence* (*data mining* and *machine learning*),
# and *optimization*; the skills are computer skills, communication skills, project and
# change management, etc.
#
# BA by itself is not a science. It is the total set of knowledge required to solve business
# problems in a rational way. Experience in BA projects and knowledge of the business area
# the data comes from (such as healthcare, advertising, or finance) is also valuable for a
# successful business analyst.
#
# BA is often subdivided into three consecutive activities: *descriptive analytics*,
# *predictive analytics*, and *prescriptive analytics*. During the descriptive phase, data is
# analyzed and patterns are found. The insights are consequently used in the predictive phase
# to predict what is likely to happen in the future, if the situation remains the same.
# Finally, in the prescriptive phase, alternative decisions are determined that change the
# situation and which will lead to desirable outcomes.
#
# :::{note} Example: Hotel Revenue Management
# :label: eg-1-1
#
# A hotel chain analyzes its reservations to look for patterns: which are the busiest days
# of the week? What is the impact of events in the city? Is there a seasonal pattern?
# (*descriptive*). The outcomes are used to make a prediction for the revenue in the
# upcoming months (*predictive*). By changing the pricing of the rooms in certain situations
# (such as sports events), the expected revenue can be maximized (*prescriptive*).
# :::

# %% [markdown]
# ## Steps of a BA Project
#
# Analytics can only start when there is data. Certain organizations already have a
# centralized *data warehouse* in which relevant current and historical data is stored for
# the purpose of reporting and analytics. Setting up such a data warehouse and maintaining
# it is part of the *business intelligence* (BI) strategy of a company. However, not all
# companies have such a centralized database, and even when it exists, it rarely contains
# all the information required for a certain analysis. Therefore, data often needs to be
# collected, cleansed and combined with other sources. Data collection, cleansing and
# further pre-processing is usually a very time-consuming task, often taking more time than
# the actual analysis.
#
# :::{note} Example: Hotel Revenue Management, Data Collection
# :label: eg-1-2
#
# In the [hotel revenue management example above](#eg-1-1) we need historical data on
# reservations but also data on historical and future events in the surroundings of the
# hotel. There are many reasons why this data can be hard to get: reservation data may only
# be stored at an aggregated level, there may have been changes in IT systems which
# overrode previously collected data, or there may be no centrally available list of events.
# Many organizations assume they already have all the data required. Only once the data
# scientist actually asks for it, say reservation data combined with the booking date, or
# the event list for the surrounding area, does the hotel find out that it is missing.
# :::
#
# Data collection and pre-processing are therefore always the first steps of a BA project.
# The data science steps proper begin after them, with descriptive analytics. A BA project
# does not end with prescriptive analytics either, that is, with generating an (optimal)
# decision. The decision still has to be implemented, which requires skills such as change
# management.
#
# To summarize, we distinguish the following steps in a BA project:
#
# :::{figure} images/lecture8_fig-steps.png
# :label: fig-ba-steps
#
# The steps of a BA project.
# :::
#
# [](#fig-ba-steps) suggests a linear process, but in practice this is rarely the case. At
# many of the steps, depending on the outcome, you might revisit earlier steps: if the
# predictions are not accurate enough for a particular application, for example, you might
# collect extra data to improve them. Not all BA projects include prescriptive analytics
# either; many projects have insight or prediction as their goal and finish after the
# descriptive or predictive steps.
#
# The major scientific fields of study corresponding to these BA steps are:
#
# :::{figure} images/lecture8_fig-fields.png
# :label: fig-ba-fields
#
# The scientific fields behind each step of a BA project.
# :::
#
# Next to cleansing, *feature engineering* is an important part of data preparation, to be
# discussed later. During descriptive analytics you get an understanding of the data: you
# visualize it and summarize it using the tool of statistical data analysis. A good
# understanding at this stage is what makes the right choices possible in the steps that
# follow.

# %% [markdown]
# ## Predictive and Prescriptive Analytics
#
# Following the descriptive analytics step, a BA project continues with predictive
# analytics. A target value that we want to predict is specified, and the parameters of the
# selected predictive method are determined based on the data available. We say that the
# model is *trained* on the data. The methods originate from inferential statistics and
# machine learning, which have their respective roots in mathematics and computer science.
# Although the approach and the background of these fields are quite different, the
# techniques largely overlap.
#
# Finally, during the prescriptive analytics phase, options are found to maximize a certain
# objective. Because the future is always unpredictable to a certain extent, optimization
# techniques often have to account for this randomness. The field that specializes in this
# is (mathematical) *optimization*. It overlaps partially with *reinforcement learning*,
# which has its roots in computer science. A special feature of reinforcement learning is
# that prediction and optimization are integrated: it combines the predictive and
# prescriptive phases in one method.
#
# :::{note} Example: Hospital Ward Planning
# :label: eg-1-3
#
# A hospital wants to improve how patients flow through its wards. It collects data on
# admissions and bed occupancy, enriched by data on staff schedules and patient diagnoses.
# After the data analysis and visualization (descriptive analytics), a method is selected
# that predicts, given a patient's characteristics and treatment plan, the probability that
# they will need intensive care or a longer stay (predictive analytics). In the prescriptive
# step, the best assignment of beds, nurses, and operating-room slots is determined.
# :::
#
# :::{note} Example: Hotel Revenue Management, Continued
# :label: eg-1-4
#
# Consider again [Example 1.2](#eg-1-2) on hotel revenue management. After having studied
# the influence of events and, for example, intra-week fluctuations on hotel reservations in
# the descriptive step, demand per price class is forecast in the predictive step. These
# forecasts are input to an optimization algorithm that determines, on a daily basis, the
# prices that maximize total revenue.
# :::

# %% [markdown]
# ## Data Science and Big Data
#
# Two terms are closely related to BA: *data science* and *big data*. Data science is an
# older term which has recently shifted in meaning and increased in popularity. It is a
# combination of different scientific fields all concerned with extracting knowledge from
# data, mainly data mining and statistics. Part of the popularity probably stems from the
# Harvard Business Review calling a data scientist role "the sexiest job of 21st century", a
# label that correctly anticipated the huge demand for data scientists that followed. The
# knowledge base of data scientists and business analysts largely overlaps. However, the
# deliverable of BA is improved business performance, whereas data scientists focus more on
# methods and insights from data. Improved business performance requires optimization to
# generate decisions and *soft skills* to implement them. Data science is more often
# associated with techniques from computer science such as machine learning; BA, in
# contrast, is more associated with mathematics and industrial engineering.
#
# Big data is a related term. It differentiates itself from regular data sets by the
# so-called *3 V's*: *volume*, *variety*, and *velocity*. A data set is considered to be
# "big data" when the amount of data is too much to be stored in a regular database, when it
# lacks a homogeneous structure (i.e., free text instead of well-described fields), and/or
# when it is only available real-time. Big data requires adapted storage systems and
# analysis techniques in order to exploit it.
#
# Big data receives a lot of attention today because of the speed at which it is collected.
# As more and more devices and sensors that automatically generate data connect to the
# internet (the *internet of things*), the amount of stored data doubles approximately
# every 3 years. However, most BA projects do not involve big data, but use relatively
# small and structured data sets. Such a data set may have originated as big data, with the
# relevant information later extracted from it.
#
# :::{note} Example: Passenger Counting from Camera Images
# :label: eg-1-5
#
# Cameras in metro stations are used to surveil passengers. Using image recognition software
# the numbers of passengers can be extracted, which can be used as input for a prediction
# method that forecasts future passenger volumes.
# :::

# %% [markdown]
# ## From Real-Life Problem to Model
#
# Every optimization problem in this course has the same three ingredients. A real-life
# problem consists of:
#
# - a *decision* we get to make, which becomes the *decision variables*;
# - a *system* that constrains what decisions are allowed, which becomes the *constraints*;
# - an *outcome* we care about, which becomes the *objective function*.
#
# The real-life question, "which feasible decision gives the best outcome?", then becomes a
# precise mathematical one. The value of this translation is that all kinds of problems,
# from very different domains, can be expressed in the same "universal modeling language"
# of mathematics, so that the same general solution methods and software apply to all of
# them.
#
# The workflow is a loop rather than a straight line:
#
# > real-life problem → *(modeling)* → model → *(solving)* → decision
#
# with data and data analysis feeding the model's parameters, and with results often
# sending you back to refine the model. Most of the effort, and most of this course, is in
# the modeling step, not the solving step.
#
# > "The formulation of a problem is often more essential than its solution, which may be
# > merely a matter of mathematical or experimental skill."
# > Albert Einstein & Leopold Infeld, *The Evolution of Physics* (1938)
#
# In [Linear Optimization](lecture8_linear-optimization.ipynb) we make this concrete with a
# four-step modeling approach: study the problem in detail, define the decision variables,
# define the objective, define the constraints.

# %% [markdown]
# ## Further Reading
#
# The sections above closely follow Section 1.1 of Koole (2019). The rest of Chapter 1 goes
# into the historical background of BA's constituent fields, a non-technical overview of
# techniques and tooling (R, Python, Excel), and the legal and ethical aspects of working
# with data; none of that is needed for the optimization and simulation material that
# follows, but it is worth reading once. Practical setup (Python, libraries, solvers) is
# covered on the [home page](../index.md#tooling).

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 1, "Introduction."
# - Wikipedia contributors. "Business analytics." *Wikipedia, The Free Encyclopedia.*
# - Davenport, T.H., & Harris, J.G. (2007). *Competing on Analytics: The New Science of
#   Winning*. Harvard Business School.
# - Einstein, A., & Infeld, L. (1938). *The Evolution of Physics*. Cambridge University
#   Press.
# - Powell, S.G., Baker, K.R., & Lawson, B. (2009). "Impact of errors in operational
#   spreadsheets." *Decision Support Systems*, 46:126–132.
