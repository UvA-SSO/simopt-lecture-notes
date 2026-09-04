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
# # Lecture 9: Applications of Integer Linear Optimization

# %% [markdown]
# In this section we discuss a number of problems that can be solved with ILO. The first is the set cover problem. We have a so-called universe $U = \{1, \dots, m\}$, and sets $S_1, \dots, S_n$, with $S_i \subset U$. We are looking for the smallest selection of sets that covers $U$.
#
# As an example, let $U$ be set of locations where incidents can happen, and every set $S_i$ the set of locations that can be reached by an ambulance from a certain base station within a certain target time. Then the set cover problem is as follows: what is the minimum number of ambulances needed and what are their base stations such that all locations can be reached within the target time?
#
# The ILO formulation is as follows:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} x_i \\
# \text{subject to} \quad & \sum_{i: u \in S_i} x_i \ge 1 \text{ for all } u \in U; \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# The binary constraints are necessary, as the following example shows. Let $U = \{1, 2, 3\}$ and $S_1 = \{1, 2\}$, $S_2 = \{1, 3\}$, and $S_3 = \{2, 3\}$. Then any combination of 2 sets is optimal but the LO relaxation has optimal value 1.5 with solution $(0.5, 0.5, 0.5)$.
#
# The main constraint is often replaced by the following more convenient notation: $\sum_{j=1}^{n} a_{uj} x_j \ge 1$ with $a_{uj} = 1$ if $u \in S_j$, 0 otherwise.
#
# > **Erratum applied (p. 98):** the summation index is $x_j$, not $x_i$ (the book prints $\sum_{j=1}^{n} a_{uj} x_i$).
#
# :::{exercise}
# :label: ex-6-12
#
# Solve the following ILO problem, inspired by Guenin, Könemann, and Tunçel (2014). A swimming pool is open during 12 hours, and the lifeguards at duty should be selected. Every lifeguard has his/her own working hours and wage, as given in the table. Select the optimal combination of lifeguards assuring at least 1 lifeguard at every moment. The hours mentioned are the first and last hour that each lifeguards works, thus Ben/Celia/Fred is a feasible solution.
#
# | Lifeguard | Ann | Ben | Celia | Dick | Estelle | Fred |
# |---|---|---|---|---|---|---|
# | Hours | 1-6 | 1-4 | 5-8 | 7-10 | 7-12 | 9-12 |
# | Wages | 8 | 6 | 6 | 3 | 7 | 3 |
#
# Solve it in Excel and use the SUMPRODUCT function and a matrix with the values of $a_{ui}$.
# :::
#
# A generalization of the set cover problem is the covering problem. Instead of having to cover each element of the universe by 1 it can be more general. This leads to the following problem formulation:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} x_i \\
# \text{subject to} \quad & \sum_{i=1}^{n} a_{ui} x_i \ge b_u \text{ for all } u \in U; \\
# & x_i \in \mathbb{N}_0 = \{0, 1, 2, \dots\} \text{ for all } i.
# \end{aligned}
# $$
#
# (shift-scheduling)=
# This problem can be applied to shift scheduling, a problem already introduced by Dantzig (1954). In shift scheduling, we have to find the best combination of shifts of employees, which in total cover the required workforce in every time interval. The seminal problem studied by Dantzig involved employees at a toll station, where the required coverage fluctuates during the day.
#
# To model this as a covering problem, let $U$ be the set of time intervals. Every set $S_i$ corresponds to a shift (with $a_{ui} = 1$ meaning shift $i$ works at time $u$), and $b_u$ the number of required workers at time $u$. Then $x_i$ corresponds to the number of employees that need to have shift $i$. By adding a coefficient to $x_i$ in the objective we can add different costs to the shifts.
#
# :::{exercise}
# :label: ex-6-13
#
# The required staffing in a call center, from 9am to 9pm in 30-minute intervals, is as follows:
#
# 10, 11, 13, 16, 16, 13, 11, 10, 10, 11, 12, 13, 14, 14, 13, 11, 10, 9, 9, 10, 9, 8, 8, 8.
#
# There are 2 types of shifts:
#
# - 8 hours working time, with a 30-minute unpaid break in the middle, wage 20 Euro/hr, possible starting times every half hour from 9am to 12:30pm;
# - 4 hours consecutive, wage 24 Euro/hr, possible starting times every half hour from 9am to 5pm.
#
# Formulate this as a covering problem and solve it with the Excel solver.
# :::
#
# Machine scheduling is another important class of ILO problems. However, because it involves some modeling tricks that are discussed under [Machine Scheduling](lecture9_advanced-modeling.ipynb#machine-scheduling), we defer discussing it to that section.
#
# The next two exercises concern problems that can be solved with appropriately chosen binary decision variables.
#
# :::{exercise}
# :label: ex-6-14
#
# For a day at a school, classes have to be assigned to professors such that each class has an hour with each required professor and such that there are no conflicts such as a professor having to teach two classes at the same time. A matrix with entries $a_{cp}$ indicates which classes need to have which professors: when $a_{cp} = 1$ then class $c$ needs to have professor $p$, otherwise $a_{cp} = 0$.
#
# Formulate an ILO model that minimizes the total number of hours that classes have to spend at school. A class remains at school until right after the last hour it has seen a professor.
# :::
#
# :::{exercise}
# :label: ex-6-15
#
# For a classroom assignment, pairs need to be made of $n$ students. Each student can give a list of students he or she is willing to work with. Formulate an ILO model that maximizes the number of pairs that can be made. Each student is only allowed to be part of one pair, but it might not be possible to assign all students to a pair.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.5 "Example ILO Problems."
# - Dantzig, G.B. (1954). "A comment on Edie's 'Traffic delays at toll booths.'" *Journal of the Operations Research Society of America*, 2(3):339–341.
# - Guenin, B., Könemann, J., & Tunçel, L. (2014). *A Gentle Introduction to Optimization*. Cambridge University Press.
