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
# description: "Homework exercises for Lecture 13 on simulation optimization."
# thumbnail: null
# ---
# # Lecture 13: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in the other Lecture 13 notebooks, from
# [About Simulation Optimization](lecture13_about-simopt.ipynb) to
# [Gradient Methods](lecture13_gradient-methods.ipynb), check what you just read. This notebook collects the larger exercises for Lecture 13: independent problems
# worth more time.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::
#
# In a simulation setting, the number of experiments $n$ is typically large enough that
# the $t$-distribution with $n - 1$ degrees of freedom is very close to the normal
# distribution. The exercises below therefore use the inverse standard normal
# distribution as an approximation for the inverse $t$-distribution, via these common
# critical values:
#
# | confidence level | critical value |
# |---|---|
# | 0.90 | 1.645 |
# | 0.95 | 1.96 |
# | 0.99 | 2.575 |

# %% [markdown]
# ## Homework Exercises

# %% [markdown]
# :::{exercise}
# :label: hw-13-1
#
# A bank has 4 investment options with returns $10, 8, 5, 12$ and investment costs
# $9, 6, 3, 10$ (all figures in thousands of euro). The bank wants to maximize total
# return within a total investment budget of 19.
#
# a. Model this as an integer linear optimization problem.
#
# Due to speculation, the investment budget is now uncertain, captured by a random
# variable $X$ that can be simulated. The bank incurs a penalty of 3 if the budget is
# exceeded, deducted from the total return.
#
# b. Describe how to simulate the expected return of a given investment policy.
#
# The bank simulates two investment policies from part b, each independently 100 times.
# The average returns are 22.5 and 21.8, with standard deviations 3.6 and 2.7,
# respectively.
#
# c. Give a 95% confidence interval for the difference in expected returns, and motivate
#    whether one policy can be concluded to be significantly better than the other.
#
# d. Describe two ways to obtain a narrower 95% confidence interval.
# :::
#

# %% [markdown]
#
# :::{solution} hw-13-1
# :label: sol-hw-13-1
# :class: dropdown
#
# a. Number the investments 1-4 and let binary $x_i = 1$ if investment $i$ is chosen. The
#    ILO is
#
#    $$
#    \begin{aligned}
#    \max \quad & 10x_1 + 8x_2 + 5x_3 + 12x_4 \\
#    \text{s.t.} \quad & 9x_1 + 6x_2 + 3x_3 + 10x_4 \le 19 \\
#    & x_i \in \{0, 1\} \text{ for all } i.
#    \end{aligned}
#    $$
#
# b. Compute the total return $R$ and budget spend $B$ of the chosen policy. Simulate a
#    budget realization $x$ from $X$: if $x < B$ the simulated return is $R - 3$, if
#    $x \ge B$ it is $R$. Repeat this simulation many times; the sample average of the
#    simulated returns approximates the expected return.
#
# c. The 95% confidence interval is
#
#    $$
#    \left[(22.5 - 21.8) - 1.96 \sqrt{\tfrac{3.6^2}{100} + \tfrac{2.7^2}{100}},\ (22.5 -
#    21.8) + 1.96 \sqrt{\tfrac{3.6^2}{100} + \tfrac{2.7^2}{100}}\right] \approx
#    [-0.18, 1.58].
#    $$
#
#    Since 0 lies in this interval, we cannot conclude with 95% confidence that one
#    policy is significantly better than the other.
#
# d. Run more simulations, or use common random numbers (the same simulated budget
#    realizations for both policies) to reduce variance.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-13-2
#
# Two scenarios for a simulation optimization problem are each simulated 400 times
# independently. Scenario A has mean 17.3 and standard deviation 8; Scenario B has mean
# 16.2 and standard deviation 6.5.
#
# a. Is there statistical evidence, at a 95% confidence level, that A differs from B?
#
# b. If you could run 200 more simulations, would you spend them on A or B? Explain.
#
# A third scenario, C, is also run 400 times, with mean 15.7 and standard deviation 12.4.
#
# c. Is there statistical evidence, at an *overall* 95% confidence level, that A is higher
#    than both B and C?
# :::
#

# %% [markdown]
#
# :::{solution} hw-13-2
# :label: sol-hw-13-2
# :class: dropdown
#
# a. The 95% confidence interval is
#
#    $$
#    \left[(17.3 - 16.2) - 1.96\sqrt{\tfrac{8^2}{400} + \tfrac{6.5^2}{400}}, (17.3 - 16.2)
#    + 1.96\sqrt{\tfrac{8^2}{400} + \tfrac{6.5^2}{400}}\right] = [0.09, 2.11].
#    $$
#
#    Since 0 is not in this interval, we can conclude with 95% confidence that A is
#    significantly different from (better than) B.
#
# b. Scenario A has the larger sample standard deviation, so adding simulations there
#    reduces the confidence interval width the most: with the extra 200 on A, the width
#    becomes approximately
#    $1.96\sqrt{\tfrac{8^2}{600} + \tfrac{6.5^2}{400}} \approx 0.90$; with them on B,
#    approximately $1.96\sqrt{\tfrac{8^2}{400} + \tfrac{6.5^2}{600}} \approx 0.94$. So the
#    extra simulations should go to A.
#
# c. Test $H_0(B, A)$: B is not worse than A, and $H_0(C, A)$: C is not worse than A, both
#    at significance level $\alpha = 1 - \sqrt{1 - 0.05} \approx 0.025$ (adjusted for
#    testing two comparisons at once), whose corresponding critical value is
#    approximately 1.96. If both null hypotheses are rejected, A is significantly better
#    than both B and C.
#
#    Since $16.2 \le 17.3 - 1.96\sqrt{6.5^2 + 8^2}/\sqrt{400} = 16.28$, $H_0(B, A)$ is
#    rejected. Since $15.7 \le 17.3 - 1.96\sqrt{12.4^2 + 8^2}/\sqrt{400} = 15.85$,
#    $H_0(C, A)$ is also rejected. So, with 95% confidence, A is better than both B and C.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-13-3
#
# Two experiments are each simulated 100 times, independently of each other. The average
# outcomes are 4.5 and 3.6, with standard deviations 2.4 and 3.7, respectively.
#
# a. Give an approximation for the distribution of the difference of the sample averages.
#
# b. Give a 95% confidence interval for the difference in expected outcomes.
#
# Now 4 experiments are each run 100 times, with average outcomes $4.5, 3.6, 2.8, 3.1$ and
# standard deviations $2.4, 3.7, 5, 2$, respectively.
#
# c. Is there statistical evidence, at a 5% significance level, that the first experiment
#    has the highest expected outcome? Use an appropriate test.
# :::
#

# %% [markdown]
#
# :::{solution} hw-13-3
# :label: sol-hw-13-3
# :class: dropdown
#
# a. Let $X, X'$ be the two experiments' outcomes, with $n = 100$ samples each, and let
#    $Y = X - X'$. By the central limit theorem,
#
#    $$
#    \frac{1}{n}\sum_{i=1}^{n} (X_i - X_i') = \frac{1}{n}\sum_{i=1}^{n} Y_i \sim N\left(EY,
#    \frac{\sigma^2(Y)}{n}\right)
#    $$
#
#    approximately, with $EY = EX - EX' \approx 4.5 - 3.6 = 0.9$ and, since $X$ and $X'$
#    are independent, $\sigma^2(Y) = \sigma^2(X) + \sigma^2(X') \approx 2.4^2 + 3.7^2 =
#    19.45$. So the difference of sample averages is approximately
#    $N(0.9, 0.1945)$: mean 0.9, standard deviation $\approx 0.44$.
#
# b. Using part a, the 95% confidence interval is
#
#    $$
#    \left[0.9 - 1.96\sqrt{\tfrac{2.4^2}{100} + \tfrac{3.7^2}{100}}, 0.9 +
#    1.96\sqrt{\tfrac{2.4^2}{100} + \tfrac{3.7^2}{100}}\right] \approx [0.04, 1.76].
#    $$
#
# c. Test the three null hypotheses that experiment 2, 3, or 4 (respectively) has a
#    *higher* outcome than experiment 1. With three simultaneous comparisons, adjust the
#    confidence level to $\sqrt[3]{0.95} \approx 0.983$, with critical value
#    approximately 2.12. All three null hypotheses must be rejected to conclude
#    experiment 1 has the significantly highest outcome.
#
#    The first test asks whether $4.5 > 3.6 + 2.12\sqrt{2.4^2 + 3.7^2}/\sqrt{10}$;
#    working out the right-hand side gives $4.5 > 4.53$, which is false, so this null
#    hypothesis cannot be rejected already. We therefore cannot conclude that experiment
#    1 has the significantly highest outcome (the other two comparisons, $4.5 > 3.97$
#    and $4.5 > 3.76$, would in fact have been rejected, but that no longer matters once
#    one comparison fails).
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-13-4
#
# In the setting of [the newsvendor exercise](lecture12_exercises.ipynb#hw-12-3), suppose
# 100 simulations are performed for every $\pi \in S = \{40, 41, 42, \dots, 60\}$: 2000
# simulations in total. For each $\pi$, the average profit over these simulations is
# calculated, and the $\pi$ with the largest average profit is returned as the best
# decision. Describe how you would spend these 2000 simulations to increase the chance of
# finding the true best $\pi$, and explain why it helps.
# :::
#

# %% [markdown]
#
# :::{solution} hw-13-4
# :label: sol-hw-13-4
# :class: dropdown
#
# Rather than splitting the whole budget of 2000 evenly across all 21 candidate values of
# $\pi$, use a two-phase ranking-and-selection approach. In an orientation phase, simulate
# every candidate a smaller number of times (e.g. 30 each, spending $30 \times 21 = 630$
# of the budget) and use a conservative $t$-test (accounting for the number of
# comparisons made) to discard candidates that are significantly worse than some other
# candidate. In a second phase, divide the remaining budget ($2000 - 630 = 1370$
# simulations) over only the candidates that were not discarded.
#
# Many problems have candidates that are clearly worse than the top few; identifying and
# discarding these early frees up simulation budget to more precisely compare the
# remaining, more promising candidates in the second phase. This concentrates statistical
# power where it matters, increasing the chance of correctly identifying the true best
# $\pi$ compared to spreading the budget uniformly over all 21 candidates from the start.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-13-5
#
# A newsvendor considers the order sizes $S = \{4, 6, 8, 10, 12\}$ and applies option 2
# of [ranking and selection](lecture13_ranking-and-selection.ipynb#option-2) with a
# budget of $m = 1400$ runs. After the first round, the following code decides which
# order sizes to discard:
#
# ```python
# import numpy as np
# from scipy import stats
#
# y = {4: 0.98, 6: 1.40, 8: 1.55, 10: 1.15, 12: 0.45}
# s = {4: 1.5, 6: 1.5, 8: 1.5, 10: 1.5, 12: 1.5}
# m0 = 100
# alpha = 0.05
# k = len(y)
# alpha_star = 1 - (1 - alpha) ** (1 / (k - 1))
# beta = stats.t.ppf(1 - alpha_star, m0 - 1)
#
# survivors = []
# for order in y:
#     discard = False
#     for other in y:
#         if other != order:
#             margin = beta * np.sqrt(s[order] ** 2 + s[other] ** 2) / np.sqrt(m0)
#             if y[order] <= y[other] - margin:
#                 discard = True
#     if not discard:
#         survivors.append(order)
# print(survivors)
# ```
#
# You may use that $\sqrt[4]{0.95} \approx 0.9873$, that `stats.t.ppf(0.9873, 99)` is
# about 2.27, that `stats.t.ppf(0.95, 99)` is about 1.66, and that
# $\sqrt{1.5^2 + 1.5^2} \approx 2.12$. The code uses the $t$-distribution, as on the
# slides. With $m_0 = 100$ runs, the normal critical values are a good approximation
# (about 2.23 and 1.64), and you may use those instead; here they give the same
# answers.
#
# a. What do `y`, `s` and `m0` stand for? Compute `alpha_star`.
#
# b. What does the code print? Show your computation.
#
# c. All standard deviations in `s` are equal. Explain why it would then be enough to
#    compare every order size only with the order size that has the highest average.
#
# d. Change the code so that, for every discarded order size, it also prints an order
#    size that beats it.
#
# e. Which line do you change to do every test at significance level 0.05, without the
#    Šidák correction? Which order sizes are discarded then, and why is this not a good
#    idea?
#
# f. Add code that computes how many extra runs each surviving order size gets in the
#    second round. What is this number here?
# :::
#

# %% [markdown]
#
# :::{solution} hw-13-5
# :label: sol-hw-13-5
# :class: dropdown
#
# a. `y` holds the average profit $y(\pi)$ of each order size after the first round, `s`
#    the sample standard deviation $s(\pi)$, and `m0` the number of runs $m_0$ of every
#    order size in the first round. With $k = 5$ order sizes,
#    $\alpha^* = 1 - \sqrt[4]{0.95} \approx 1 - 0.9873 = 0.0127$.
#
# b. `beta` is about 2.27, and since all standard deviations are equal, every comparison
#    uses the same margin $2.27 \times 2.12 / \sqrt{100} \approx 0.48$. An order size is
#    discarded if its average is at least 0.48 below the average of another order
#    size. The highest average is 1.55 (order size 8), so order sizes with an average
#    at most $1.55 - 0.48 = 1.07$ are discarded: 4 (0.98) and 12 (0.45). The code prints
#    `[6, 8, 10]`.
#
# c. With equal standard deviations the margin is the same for every pair. If an order
#    size is beaten by some other order size, so $y(\pi) \le y(\pi') - \text{margin}$,
#    then it is certainly beaten by the order size with the highest average, because
#    that average is at least $y(\pi')$. So comparing with that one order size gives
#    the same result.
#
# d. Remember the order size that beats it, and print it after the inner loop:
#
#    ```python
#    for order in y:
#        discard = False
#        for other in y:
#            if other != order:
#                margin = beta * np.sqrt(s[order] ** 2 + s[other] ** 2) / np.sqrt(m0)
#                if y[order] <= y[other] - margin:
#                    discard = True
#                    beaten_by = other
#        if discard:
#            print(order, "is beaten by", beaten_by)
#        else:
#            survivors.append(order)
#    ```
#
# e. Replace the line `beta = stats.t.ppf(1 - alpha_star, m0 - 1)` by
#    `beta = stats.t.ppf(1 - alpha, m0 - 1)`. Then `beta` is about 1.66, the margin is
#    $1.66 \times 2.12 / 10 \approx 0.35$, and order sizes with an average of at most
#    $1.55 - 0.35 = 1.20$ are discarded: 4, 10 and 12. This is not a good idea because
#    each order size is tested against four others: even if it is the best one, the
#    probability that at least one of the four tests discards it by chance is then
#    larger than 5%.
#
# f. The first round used $k m_0 = 5 \times 100 = 500$ runs, so 900 runs are left:
#
#    ```python
#    budget = 1400
#    n_each = (budget - k * m0) // len(survivors)
#    print(n_each)
#    ```
#
#    With three survivors, each gets 300 extra runs.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.1, §8.2, §8.3.
