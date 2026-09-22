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
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_introduction.ipynb)

# %% [markdown]
# This part of the course is about optimization and simulation: turning data and
# predictions into good decisions. This first notebook places that in context: what kind of
# analytics it is, how it relates to data science, and how a real-life problem becomes a
# mathematical model. The technical material starts in
# [Linear Optimization](lecture8_linear-optimization.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - place prescriptive analytics among the descriptive/diagnostic/predictive/prescriptive
#   sequence, and explain what distinguishes it;
# - relate business analytics to data science and to its roots in statistics, AI, and
#   operations research;
# - describe the steps of a BA project, from data collection to implementation, and which
#   field of study each step draws on;
# - translate a real-life decision problem into the decision/constraints/objective
#   ingredients of a mathematical model.

# %% [markdown]
# ## Prescriptive Analytics and Optimization
#
# Business analytics is usually split into a sequence of activities, each answering a
# different question about a business process:
#
# - *descriptive* analytics: what happened? (reporting, visualization, summary statistics);
# - *diagnostic* analytics: why did it happen?;
# - *predictive* analytics: what is likely to happen? (forecasting, machine learning);
# - *prescriptive* analytics: what should we do about it?
#
# This course is about the last step. Prescriptive analytics determines which decision leads
# to the best outcome, given the data and the predictions produced by the earlier steps. Its
# main tool is mathematical optimization, and that is what "the simulation and optimization
# part" of this course is about.
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
# ## What Is Business Analytics and Data Science?
#
# According to Wikipedia, "business analytics refers to the skills, technologies, and
# practices for continuous iterative exploration and investigation of past business
# performance to gain insight and drive business planning." In short, **business analytics
# (BA)** is a rational, fact-based approach to decision making: it is the science and the
# skills needed to turn data into decisions. BA is not itself a science. It is the total set
# of knowledge, drawn from several fields, required to solve business problems this way. It
# grew out of three older fields: statistics (drawing conclusions from data under
# uncertainty), operations research (mathematical optimization of decisions, born out of
# military and industrial logistics), and artificial intelligence (algorithms that learn
# patterns from data).
#
# **Data science** overlaps heavily with business analytics but is usually the broader term.
# It also covers domains outside business (science, government, healthcare) and puts
# relatively more weight on handling large-scale, unstructured data (text, images, sensor
# streams) alongside the structured data (rows and columns in a database) that classical
# statistics and optimization assume. BA's deliverable is, more specifically, improved
# business performance, which requires optimization and soft skills on top of what data
# science delivers.
#
# Working effectively in this field takes both depth in one specialism (statistics,
# optimization, or software engineering) and enough breadth to talk to the other
# specialisms and to the business, often called being a "T-shaped" analyst: one deep skill,
# broad awareness. With two deep skills, the analogy becomes "Π-shaped". Organizations
# differ widely in how much they have adopted analytics, from purely descriptive reporting
# to systematic prescriptive decision-making, and that analytics maturity is itself
# something organizations assess and try to grow. Finally, working with data responsibly
# raises legal and ethical questions (privacy, bias, fairness) that sit alongside the
# technical material throughout.

# %% [markdown]
# ## The Steps of a BA Project
#
# A BA project runs through six steps: data collection, data pre-processing, descriptive
# analytics, predictive analytics, prescriptive analytics, and implementation. The middle
# four are the *data science steps*; the full sequence, from raw data to a change in how the
# organization operates, is the *full BA project*.
#
# :::{figure} images/lecture8_fig-steps.png
# :label: fig-ba-steps
#
# The steps of a BA project.
# :::
#
# [](#fig-ba-steps) suggests a straight line, but in practice a project loops back
# constantly. Disappointing predictions, for example, may send you back to collect more
# data. Briefly, per step:
#
# - **Data collection.** Analytics can only start once there is data. Some organizations
#   keep a centralized *data warehouse* for this, as part of their *business intelligence*
#   (BI) strategy. Where that is missing or incomplete, data must be collected and combined
#   from elsewhere, often the most time-consuming part of a project.
# - **Data pre-processing.** Raw data is cleaned (missing or impossible values handled)
#   and, through *feature engineering*, combined into more informative attributes, e.g.
#   turning "day of week" and "time" into "business hours".
# - **Descriptive analytics.** The data is explored and summarized through visualization,
#   clustering, and hypothesis testing (see [Variability](lecture12_variability-recap.ipynb))
#   to build understanding before modeling. There is no target value here, so this is also
#   called *unsupervised learning*.
# - **Predictive analytics.** A target value (e.g. sales, or whether a customer pays) is
#   predicted from the data; the model is *trained* on historical data for which that target
#   is already known. This is also called *supervised learning*, in the flavors *regression*
#   (numerical target) and *classification* (categorical target).
# - **Prescriptive analytics.** Given the predictions, the decision that maximizes (or
#   minimizes) an objective is found: the subject of this course, using (mathematical)
#   optimization, possibly combined with prediction in *reinforcement learning*.
# - **Implementation.** The decision has to be put into practice, which needs skills well
#   beyond data science: change management, communication, and project management.
#
# Each step leans on a different field of study:
#
# :::{figure} images/lecture8_fig-fields.png
# :label: fig-ba-fields
#
# The scientific fields behind each step of a BA project.
# :::
#
# [](#fig-ba-techniques) lists, per data-science step, some of the specific techniques used.
# Several reappear later in this course: clustering and hypothesis testing in
# [Variability](lecture12_variability-recap.ipynb), simulation in
# [Simulation](lecture12_simulation.ipynb), and linear optimization, our main prescriptive
# technique, starting in [Linear Optimization](lecture8_linear-optimization.ipynb). Dynamic
# programming and reinforcement learning are prescriptive techniques as well, but are not
# covered in this course.
#
# :::{figure} images/lecture8_fig1.1.png
# :label: fig-ba-techniques
#
# An overview of the most-often used data science techniques.
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
# This notebook only sketches the non-technical background. Koole (2019), Chapter 1, goes
# deeper into all of it, including "big data", organizational analytics-maturity models in
# more detail, and the legal and ethical aspects of working with data. None of that is
# needed for the optimization and simulation material that follows, but it is worth reading
# once. Practical setup (Python, libraries, solvers) is covered on the
# [course home page](../index.md#tooling), not repeated here.

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
