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
# description: "The probability and statistics that Lectures 12 and 13 use: random variables, expectation and variance, the law of large numbers, the central limit theorem, confidence intervals and hypothesis tests."
# thumbnail: null
# ---
# # Lecture 12: Variability (Recap)
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture12_variability-recap.ipynb)

# %% [markdown]
# Simulation is about variability: the inputs of a model are random, so its output is random too. This notebook recaps the probability and statistics from the statistics part of the course that Lectures 12 and 13 use. It is a reference, not new material, so skim what you already know. The simulation material starts in [Why Simulate?](lecture12_why-simulation.ipynb). How to sample random numbers and compute these quantities in Python follows in [Sampling a Random Variable](lecture12_sampling.ipynb) and [Monte Carlo Simulation](lecture12_monte-carlo.ipynb).
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - distinguish discrete and continuous random variables and describe them with a pmf, pdf or cdf
# - compute the expectation and variance of a random variable, and of sums and averages of random variables
# - explain the law of large numbers and the central limit theorem, and why simulation relies on them
# - construct and interpret a confidence interval for an expectation
# - explain how a hypothesis test compares two expectations

# %% [markdown]
# ## Random Variables
#
# A **random variable** (RV) $X$ models an uncertain outcome, such as the duration of an activity or the number of customers that arrive in an hour. Random variables come in two types.
#
# A **discrete** RV takes a finite or countable number of values, for example the sum of two dice or a number of arrivals ($0, 1, 2, \dots$). It is described by its probability mass function (pmf) $P(X = x)$. For the sum of two dice, 6 of the 36 equally likely outcomes give 7, so $P(X = 7) = 6/36 = 1/6$.
#
# A **continuous** RV can take any value in a range, for example a duration. Every single value then has probability 0: nobody is exactly 1.80 m tall when measured to ten decimals. Instead, a probability density function (pdf) $f(x)$ gives probabilities of intervals, $P(a \le X \le b) = \int_a^b f(x)\,dx$. Where the density is high, outcomes are more likely.
#
# Both types have a cumulative distribution function (cdf) $F(x) = P(X \le x)$. It is non-decreasing, goes from 0 to 1, and is a step function for a discrete RV. For $X$ uniform on $[a, b]$, every subinterval of the same length is equally likely and
#
# $$
# F(x) = \frac{x - a}{b - a} \quad \text{for } a \le x \le b.
# $$
#
# Two RVs are **independent** if the outcome of one does not change the probabilities of the other. Repeated experiments, such as simulation runs, are usually independent; activities that depend on the same weather or the same interest rate are not.
#
# :::{exercise}
# :label: ex-3-3
#
# For the cdf below, give the possible outcomes and their probabilities.
#
# ![The cdf of a discrete random variable](images/lecture12_ex3.3-cdf.png)
# :::

# %% [markdown]
# (common-distributions)=
# ## Common Distributions
#
# The table lists the distributions used in Lectures 12 and 13, with their expectation and variance (defined in the next section).
#
# | distribution | type | expectation | variance | typical use |
# |---|---|---|---|---|
# | Bernoulli($p$): 1 with probability $p$, else 0 | discrete | $p$ | $p(1-p)$ | a yes/no outcome |
# | binomial($n$, $p$): number of 1s in $n$ independent Bernoulli($p$) trials | discrete | $np$ | $np(1-p)$ | number of successes |
# | Poisson($\lambda$) | discrete | $\lambda$ | $\lambda$ | number of arrivals in a period |
# | uniform[$a$, $b$] | continuous | $(a+b)/2$ | $(b-a)^2/12$ | a value somewhere between $a$ and $b$ |
# | exponential with rate $\lambda$: $F(x) = 1 - e^{-\lambda x}$, $x \ge 0$ | continuous | $1/\lambda$ | $1/\lambda^2$ | time between arrivals |
# | normal $N(\mu, \sigma^2)$ | continuous | $\mu$ | $\sigma^2$ | sums and averages (see the CLT below) |
# | lognormal($\mu$, $\sigma$): $e^Y$ with $Y \sim N(\mu, \sigma^2)$ | continuous | $e^{\mu + \sigma^2/2}$ | $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$ | durations: positive and skewed to the right |
#
# Watch the parameters, because they are not the same everywhere. $N(3, 4)$ has variance 4 and standard deviation 2. An exponential distribution is described by its rate $\lambda$ or by its mean $1/\lambda$. The parameters $\mu$ and $\sigma$ of a lognormal distribution are those of the underlying normal distribution, not its own mean and standard deviation: lognormal(1, 1) has expectation $e^{1.5} \approx 4.48$.

# %% [markdown]
# ## Expectation and Variance
#
# The **expectation** $EX$ is the average outcome, weighted with the probabilities:
#
# $$
# EX = \sum_x x \, P(X = x) \quad \text{(discrete)}, \qquad EX = \int x f(x)\,dx \quad \text{(continuous)}.
# $$
#
# For a die, $EX = \frac16 (1 + 2 + \dots + 6) = 3.5$. For a function $r$ of $X$, the expectation $E[r(X)]$ weights the values $r(x)$ in the same way, $E[r(X)] = \sum_x r(x) P(X = x)$ (or $\int r(x) f(x)\,dx$). Simulation is mostly about estimating such an $E[r(X)]$.
#
# The **variance** $\sigma^2(X) = E[(X - EX)^2]$ measures the spread of $X$ around $EX$: the expected squared distance to the expectation. For a die, $\sigma^2(X) = \frac16 \big((1-3.5)^2 + \dots + (6-3.5)^2\big) = 35/12 \approx 2.92$. The **standard deviation** (SD) $\sigma(X) = \sqrt{\sigma^2(X)}$ has the same unit as $X$, which makes it easier to interpret; the variance has easier rules, which follow now.
#
# (sums-of-rvs)=
# ### Sums and Averages
#
# For RVs $X$ and $Y$ and a constant $a$:
#
# $$
# E(X + Y) = EX + EY, \qquad E(aX) = aEX,
# $$
#
# whether or not $X$ and $Y$ are independent, and
#
# $$
# \sigma^2(X + Y) = \sigma^2(X) + \sigma^2(Y) \text{ if } X \text{ and } Y \text{ are independent}, \qquad \sigma^2(aX) = a^2 \sigma^2(X).
# $$
#
# Note that the variances add up, not the SDs.
#
# :::{note} Example: Two Business Units
# :label: eg-3-4
#
# The profits $X$ and $Y$ of two business units next year have $EX = 8$, $\sigma(X) = 3$, $EY = 12$ and $\sigma(Y) = 4$ (in M€). The total expected profit is $E(X + Y) = 20$. If $X$ and $Y$ are independent, then
#
# $$
# \sigma(X + Y) = \sqrt{3^2 + 4^2} = 5,
# $$
#
# less than $3 + 4 = 7$: a loss in one unit is often compensated by the other. This is why investors spread their risk. It only works when the units are (close to) independent; in an economic downturn both profits tend to go down together.
# :::
#
# Now let $X_1, \dots, X_n$ be independent with the same distribution as $X$, for example the outputs of $n$ simulation runs, and let $\bar X = (X_1 + \dots + X_n)/n$ be their average. The rules above give
#
# $$
# E \bar X = \frac{EX_1 + \dots + EX_n}{n} = EX, \qquad \sigma^2(\bar X) = \frac{\sigma^2(X_1) + \dots + \sigma^2(X_n)}{n^2} = \frac{\sigma^2(X)}{n},
# $$
#
# so $\sigma(\bar X) = \sigma(X)/\sqrt n$.

# %% [markdown]
# (lln)=
# ## Law of Large Numbers
#
# The average $\bar X$ is centered on $EX$, and its variance $\sigma^2(X)/n$ goes to 0 as $n$ grows. So for large $n$ the average is close to the expectation, with high probability. This is the **law of large numbers** (LLN). It is the reason that simulation works: the average output of many independent simulation runs approaches the expected output.
#
# The SD of the average decreases with $\sqrt n$, which grows slowly: to make the SD of $\bar X$ twice as small, we need four times as many observations.

# %% [markdown]
# (clt)=
# ## Central Limit Theorem
#
# The **central limit theorem** (CLT) adds to the LLN what the distribution of the average looks like: for large $n$, $\bar X$ is approximately normal,
#
# $$
# \bar X \approx N\left(EX, \frac{\sigma^2(X)}{n}\right),
# $$
#
# whatever the distribution of $X$ itself (as long as its variance is finite). A rule of thumb is that $n \ge 30$ is large enough. [](#fig-clt) shows the CLT at work for averages of uniform[0, 1] samples: the average of a single sample is uniform, but the average of 16 samples already has the bell shape of a normal density.

# %% tags=["remove-cell"] label="clt-averages"
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEXT_COLOR = "#111827"
rng = np.random.default_rng(1)
clt_fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[f"average of {k}" for k in (1, 2, 4, 16)],
    vertical_spacing=0.2,
)
for panel, n_avg in enumerate([1, 2, 4, 16]):
    averages = rng.uniform(0, 1, (1000, n_avg)).mean(axis=1)
    clt_fig.add_trace(
        go.Histogram(
            x=averages,
            xbins={"start": 0, "end": 1, "size": 0.05},
            marker={"color": "#1f77b4", "line": {"width": 0}},
            showlegend=False,
        ),
        row=panel // 2 + 1,
        col=panel % 2 + 1,
    )
clt_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": TEXT_COLOR, "size": 13},
    margin={"l": 40, "r": 20, "t": 40, "b": 30},
    height=420,
    bargap=0.05,
)
clt_fig.update_xaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.25)")
clt_fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
clt_fig.show(config={"displayModeBar": False, "staticPlot": True})

# %% [markdown]
# :::{figure} #clt-averages
# :label: fig-clt
#
# Histograms of 1000 averages of 1, 2, 4 and 16 uniform[0, 1] samples.
# :::
#
# For a normal distribution, about 68% of the probability lies within one SD of the expectation and about 95% within two SDs (more precisely, 1.96). So with probability about 95%, the average $\bar X$ lies within $2\sigma(X)/\sqrt n$ of $EX$. Do not apply this rule to the data itself, whose distribution need not be normal; it is the average that is approximately normal.

# %% [markdown]
# (confidence-intervals)=
# ## Confidence Intervals
#
# The CLT tells us how far the average $\bar X$ of $n$ observations is likely to be from the unknown $EX$. The SD $\sigma(X)$ is usually unknown too, so we estimate it with the **sample standard deviation**
#
# $$
# S = \sqrt{\frac{\sum_{i=1}^n (X_i - \bar X)^2}{n - 1}}.
# $$
#
# Dividing by $n - 1$ instead of $n$ makes $S^2$ an unbiased estimator of $\sigma^2(X)$. A 95% **confidence interval** (CI) for $EX$ is then
#
# $$
# \left[\bar X - \frac{2S}{\sqrt n}, \ \bar X + \frac{2S}{\sqrt n}\right].
# $$
#
# The 2 is a rounded 1.96. Strictly speaking, using $S$ instead of $\sigma(X)$ calls for Student's $t$-distribution with $n - 1$ degrees of freedom, but for $n$ of 30 or more the difference is negligible. The width of the CI shrinks with $\sqrt n$: to halve it, we need four times as many observations.
#
# :::{note} Interpretation of a CI
# A 95% CI does not mean that $EX$ lies in the interval with probability 95%: $EX$ is a fixed (unknown) number, which is either in a given interval or not. What is random is the interval. If we repeated the experiment many times and computed a CI every time, about 95% of those intervals would contain $EX$. For one computed interval we say that we are 95% confident that it contains $EX$.
# :::
#
# For example, 100 measurements with average 178 and sample SD 5 give the 95% CI $[178 - 2 \times 5/\sqrt{100}, \ 178 + 2 \times 5/\sqrt{100}] = [177, 179]$.

# %% [markdown]
# (hypothesis-testing)=
# ## Hypothesis Tests
#
# A hypothesis test checks whether the data give enough evidence against a **null hypothesis** $H_0$, for example that two expectations are equal. We compute the **p-value**: the probability, if $H_0$ were true, of an outcome at least as extreme as the one observed. If the p-value is below the significance level, usually 5%, we reject $H_0$. If it is not, we cannot conclude anything: not rejecting $H_0$ is not evidence that $H_0$ is true.
#
# A test is **two-sided** if deviations in both directions count against $H_0$ (for example $H_1: EX \ne EY$) and **one-sided** if only one direction counts ($H_1: EX > EY$). For a test statistic $T$ that is approximately $N(0, 1)$ under $H_0$, a two-sided test at 5% rejects when $|T| > 1.96$ and a one-sided test rejects when $T > 1.64$.
#
# Lecture 13 compares two expectations $EX$ and $EY$, for example the expected profits of two decisions, using $n$ observations of each. There are two settings.
#
# **Independent samples.** The observations of $X$ and $Y$ are independent. With averages $\bar X$, $\bar Y$ and sample SDs $S_X$, $S_Y$, the difference $\bar X - \bar Y$ has SD $\sqrt{\sigma^2(X)/n + \sigma^2(Y)/n}$, and
#
# $$
# T = \frac{\bar X - \bar Y}{\sqrt{S_X^2/n + S_Y^2/n}}
# $$
#
# is approximately $N(0, 1)$ if $EX = EY$. Equivalently: we reject $EX = EY$ (two-sided, at 5%) exactly when the 95% CI $\bar X - \bar Y \pm 2\sqrt{S_X^2/n + S_Y^2/n}$ for $EX - EY$ does not contain 0.
#
# **Matched pairs.** The observations come in pairs $(X_i, Y_i)$ that belong together, such as the incomes of two partners in one household, or two decisions simulated under the same circumstances. Then we work with the differences $D_i = X_i - Y_i$ and make a CI for $ED = EX - EY$ from them, exactly as in the previous section. If $X_i$ and $Y_i$ are positively correlated, the differences vary less than in the independent setting, which gives a narrower CI. Lecture 13 makes use of this.
#
# :::{note} Example: Two Order Sizes
# :label: eg-two-sample-test
#
# A shop simulates its daily profit for two order sizes, 100 independent days each. Order size 1 gives an average profit of 520 with sample SD 60, order size 2 an average of 500 with sample SD 80. Then
#
# $$
# T = \frac{520 - 500}{\sqrt{60^2/100 + 80^2/100}} = \frac{20}{10} = 2 > 1.96,
# $$
#
# so at significance level 5% we conclude that the expected profits differ. The 95% CI for the difference, $20 \pm 2 \times 10 = [0, 40]$, just touches 0, in line with $T$ being just about 1.96.
# :::

# %% [markdown]
# ## Exercises
#
# These exercises can be done on paper, with a calculator.
#
# :::{exercise}
# :label: ex-recap-die
#
# Let $X$ be the outcome of a die roll and $Y$ the sum of 10 independent die rolls.
#
# a. Give $P(X \le 2)$ and $F(2.5)$, with $F$ the cdf of $X$.
#
# b. Compute $EY$ and $\sigma(Y)$, using $EX = 3.5$ and $\sigma^2(X) = 35/12$.
#
# c. Is $\sigma(Y)$ equal to $10\,\sigma(X)$? Explain.
# :::
#
# :::{exercise}
# :label: ex-3-16
#
# A hospital performs knee surgery in one of its operating rooms. An operation takes on average 50 minutes with an SD of 20 minutes, including cleaning and changing. A session consists of 8 independent operations, and a block of 7 hours is reserved for it. Use a normal approximation to estimate the probability that the 8 operations take longer than the reserved time. Use that $P(Z \le 0.35) \approx 0.64$ for $Z \sim N(0, 1)$.
# :::
#
# :::{exercise}
# :label: ex-3-21
#
# A sample of 200 observations has average 9.8 and sample SD 5.4. Give a 95% CI for the expectation, and explain what it means.
# :::
#
# :::{exercise}
# :label: ex-3-22
#
# You roll a die 12 times and get no 6.
#
# a. How many 6s did you expect?
#
# b. The probability of no 6 in 12 rolls of a fair die is $(5/6)^{12} \approx 0.11$. Can you conclude, at significance level 5%, that the die is biased? Can you conclude that it is fair?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 3, "Variability" (recap).
# - Ross, S.M. (2002). *A First Course in Probability*, 6th ed. Prentice Hall.
# - Triola, M.F. (2017). *Elementary Statistics*, 13th ed. Pearson.
