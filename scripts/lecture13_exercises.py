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
# # Lecture 13: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in
# [Simulation Optimization](lecture13_simulation-optimization.ipynb) check what you just
# read. This notebook collects the larger exercises for Lecture 13: independent problems
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
#    \max 10x_1 + 8x_2 + 5x_3 + 12x_4 \quad \text{s.t.} \quad 9x_1 + 6x_2 + 3x_3 + 10x_4
#    \le 19,\ x_i \in \{0, 1\}.
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
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.1, §8.2, §8.3.
