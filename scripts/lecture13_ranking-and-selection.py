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
# description: "Ranking and selection for a small, discrete set of solutions: splitting the simulation budget equally, or discarding bad solutions early with corrected one-sided tests."
# thumbnail: null
# ---
# # Lecture 13: Ranking and Selection
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_ranking-and-selection.ipynb)

# %% [markdown]
# [Comparing Scenarios](lecture13_comparing-scenarios.ipynb) chose between two solutions. This notebook covers the second of the [four types of problems](lecture13_about-simopt.ipynb#four-types): a small, discrete set of solutions. We discuss two ways to spend the simulation budget, an equal split over all solutions and a smarter one that discards bad solutions early, and apply both to the newsvendor problem.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate the newsvendor problem as a simulation optimization problem
# - apply ranking and selection with an equal split of the simulation budget, and explain its drawbacks
# - formulate the one-sided tests that discard solutions, and explain the Šidák correction for multiple testing
# - apply ranking and selection with discarding, and read and adapt the Python code that does so

# %% [markdown]
# (ranking-and-selection)=
# ## The Setting
#
# We solve the simulation optimization problem
#
# $$
# \max_{\pi \in S} \ E[r(X, \pi)],
# $$
#
# where $E[r(X, \pi)]$ can only be evaluated by simulation. In this notebook $S$ is discrete and relatively small, with $k = |S|$ solutions: small enough that our simulation budget of $m$ runs suffices to simulate every solution a couple of times. The task is to select the best solution, and on the way we rank the solutions by their estimated performance, hence the name **ranking and selection**.

# %% [markdown]
# (newsvendor)=
# ## The Newsvendor Problem
#
# The [newsvendor problem](lecture13_about-simopt.ipynb#eg-8-2) is a typical ranking and selection problem. A newsvendor buys $\pi$ newspapers in the morning for a price $c$ each, and sells them during the day for a price $p > c$. Demand $X$ is random, and newspapers that are left at the end of the day are worthless. The newsvendor sells $\min(X, \pi)$ papers, so the profit is
#
# $$
# r(X, \pi) = p \min(X, \pi) - c\pi,
# $$
#
# and the newsvendor chooses $\pi$ from $S = \{0, 1, \dots, K\}$, where $K$ is the number of papers that fit on the shelf. The [worked example in Comparing Scenarios](lecture13_comparing-scenarios.ipynb#crn-example) was a newsvendor problem with only two order sizes, 8 and 12.
#
# :::{note} Example: Newsvendor with 21 Order Sizes
# :label: eg-8-3
#
# A newsvendor buys newspapers for €0.75 and sells them for €1. Demand $X$ is Poisson distributed with mean 10, and there is room for 20 newspapers on the shelf. The simulation optimization problem is
#
# $$
# \max_{\pi \in S} \ E[\min(X, \pi)] - 0.75\pi, \quad S = \{0, 1, \dots, 20\},
# $$
#
# with $k = 21$ solutions. Replacing $X$ by its mean gives $\max_{\pi \in S} \min(10, \pi) - 0.75\pi$, with solution $\pi = 10$; by the flaw of averages, this need not be the best solution of the actual problem.
# :::
#
# In the code below, `simulate` simulates the profit of one order size `n` times:

# %%
import numpy as np

price, cost = 1, 0.75
mean_demand = 10
orders = np.arange(21)  # S = {0, 1, ..., 20}


def profit(order, demand):
    return price * np.minimum(demand, order) - cost * order


def simulate(order, n, rng):
    """Simulated profits of n days with the given order size."""
    return profit(order, rng.poisson(mean_demand, n))


# %% [markdown]
# For this small problem the expected profit can also be computed exactly, with $E[\min(X, \pi)] = \sum_{j=0}^{\pi-1} P(X > j)$. Real simulation optimization problems have no such formula; we only use it to check how well the methods below do.

# %%
from scipy import stats

expected_sales = [
    stats.poisson.sf(np.arange(o), mean_demand).sum() for o in orders
]
true_value = price * np.array(expected_sales) - cost * orders
best_order = orders[true_value.argmax()]
print(f"best order size: {best_order}")
print(f"expected profit of order size {best_order}: {true_value.max():.3f}")
print(f"expected profit of order size 10: {true_value[10]:.3f}")

# %% [markdown]
# Ordering 8 papers is best, not 10: with leftovers worthless and a small margin of €0.25 per sold paper, the newsvendor should order less than the mean demand.

# %% [markdown]
# (option-1)=
# ## Option 1: Split the Budget Equally
#
# The natural first approach is to simulate every solution $m/k$ times, estimate $E[r(X, \pi)]$ for each $\pi$ by the average $y(\pi)$ of its runs, and return the solution with the highest average. With the sample variance $s^2(\pi)$ we can also give a [95% CI](lecture12_variability-recap.ipynb#hypothesis-testing) for each $E[r(X, \pi)]$.
#
# For the newsvendor, a budget of $m = 4200$ runs gives 200 runs per order size:

# %%
rng = np.random.default_rng(1)
budget = 4200
runs_per_order = budget // len(orders)

means = np.zeros(len(orders))
sds = np.zeros(len(orders))
for idx, order in enumerate(orders):
    profits = simulate(order, runs_per_order, rng)
    means[idx] = profits.mean()
    sds[idx] = profits.std(ddof=1)
half_widths = 1.96 * sds / np.sqrt(runs_per_order)

best = means.argmax()
print(f"selected order size: {orders[best]}")
print(f"estimated expected profit: {means[best]:.3f}")

# %% tags=["remove-cell"]
import plotly.graph_objects as go

TEXT_COLOR = "#111827"
PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": TEXT_COLOR, "size": 13},
    "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
    "height": 360,
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.2,
    },
}
AXIS_STYLE = {"gridcolor": "rgba(128,128,128,0.25)"}
# static pictures: no hover, zoom or drag
PLOT_CONFIG = {"displayModeBar": False, "staticPlot": True}


def add_cis(fig, x, y, half_width, color, name):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"color": color, "size": 7},
            error_y={
                "type": "data",
                "array": half_width,
                "color": color,
                "thickness": 2,
                "width": 4,
            },
            name=name,
        )
    )


def finish_ci_plot(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="order size π", dtick=2, **AXIS_STYLE)
    fig.update_yaxes(title="expected profit (€)", **AXIS_STYLE)
    fig.show(config=PLOT_CONFIG)


# %% tags=["remove-cell"] label="option1-cis"
option1_fig = go.Figure()
add_cis(option1_fig, orders, means, half_widths, "#1f77b4", "95% CI")
option1_fig.add_trace(
    go.Scatter(
        x=orders,
        y=true_value,
        mode="lines",
        line={"color": "grey", "dash": "dash", "width": 2},
        name="true expected profit",
    )
)
finish_ci_plot(option1_fig)

# %% [markdown]
# :::{figure} #option1-cis
# :label: fig-option1-cis
#
# 95% CIs for the expected profit of every order size, each based on 200 runs, with the true expected profit.
# :::
#
# [](#fig-option1-cis) shows the CIs. The CIs for small order sizes are narrow: with few papers the newsvendor almost always sells everything, so the profit hardly varies. Around the optimum, the CIs of order sizes 7, 8 and 9 overlap: the budget is not enough to tell them apart. At the same time, a large part of the budget went to order sizes 14 to 20, whose CIs lie far below the others after a few runs already.
#
# Because of the noise, option 1 shows both problems mentioned in [About Simulation Optimization](lecture13_about-simopt.ipynb#simopt-challenges): it may select the wrong order size, and the estimated profit of the selected order size tends to be too high. To see how often this happens, we repeat option 1 1000 times:


# %%
def option_1(budget, rng):
    """Equal split: return the selected order size and its estimate."""
    n = budget // len(orders)
    estimates = [simulate(order, n, rng).mean() for order in orders]
    best = int(np.argmax(estimates))
    return orders[best], estimates[best]


rng = np.random.default_rng(2)
results_1 = [option_1(budget, rng) for _ in range(1000)]
selected_1 = np.array([order for order, _ in results_1])
overestimate = [est - true_value[order] for order, est in results_1]
print(f"fraction best order size selected: {np.mean(selected_1 == 8):.2f}")
print(f"average overestimation of the profit: {np.mean(overestimate):.3f}")

# %% [markdown]
# Option 1 selects the best order size 8 only about half of the time; otherwise it mostly selects 7 or 9. And the estimate of the selected order size's expected profit is too high on average, because we select the order size whose estimate happened to come out high.
#
# (crn-ranking)=
# ### Common Random Numbers
#
# The code above draws new demands for every order size. As in [Comparing Scenarios](lecture13_comparing-scenarios.ipynb#crn), we can also use common random numbers: simulate one set of demands and evaluate every order size on the same demands. This does not make the individual CIs narrower, but it reduces the variance of the differences between order sizes, and these differences decide which order size is selected. In a discrete-event simulation it is the same: reuse the same customer arrival moments for every candidate solution.
#
# :::{exercise}
# :label: ex-crn-ranking
#
# Change the code of option 1 above so that it uses common random numbers: draw `runs_per_order` demands once, before the loop, and use them for every order size.
#
# a. Compare the CIs with [](#fig-option1-cis). Are they narrower? Does the line through the averages look different?
#
# b. Change `option_1` in the same way and repeat the 1000 experiments. How often is order size 8 selected now?
#
# c. With common random numbers, $y(\pi)$ and $y(\pi')$ are positively correlated. Why does that make the discard rule of option 2 below too strict, and what would you use instead of $s^2(\pi) + s^2(\pi')$?
# :::

# %% [markdown]
# (option-2)=
# ## Option 2: Discard Bad Solutions Early
#
# Option 1 spends as many runs on clearly bad solutions as on promising ones. Option 2 first simulates all solutions a limited number of times, discards the solutions that are significantly worse than another solution, and spends the rest of the budget on the remaining ones. In general terms, with $k = |S|$ solutions and a budget of $m$ runs:
#
# 1. Simulate every solution $m_0$ times, where $m_0 < \lfloor m/k \rfloor$ is the number of runs per solution in this first round ($\lfloor \cdot \rfloor$ rounds down to an integer). Compute the average $y(\pi)$ and the sample variance $s^2(\pi)$ of each solution.
# 2. For every pair of solutions $(\pi, \pi')$, test whether $\pi$ is significantly worse than $\pi'$.
# 3. Discard every solution that is significantly worse than at least one other solution. The solutions that remain form the set $I$.
# 4. Divide the remaining budget $m - k m_0$ evenly over the solutions in $I$, and return the solution in $I$ with the highest average.
#
# (discard-test)=
# ### When to Discard a Solution
#
# Consider first one pair of solutions $\pi$ and $\pi'$, each simulated $m_0$ times, independently. We maximize, so $\pi$ is worse than $\pi'$ if $E[r(X, \pi)] < E[r(X, \pi')]$. We only discard $\pi$ if the data give strong evidence for this, so this is the alternative hypothesis of a [one-sided test](lecture12_variability-recap.ipynb#hypothesis-testing):
#
# $$
# \begin{aligned}
# H_0(\pi, \pi') &: E[r(X, \pi)] \ge E[r(X, \pi')] \quad (\pi \text{ is not worse than } \pi'), \\
# H_1(\pi, \pi') &: E[r(X, \pi)] < E[r(X, \pi')] \quad (\pi \text{ is worse than } \pi').
# \end{aligned}
# $$
#
# The difference of the averages $y(\pi') - y(\pi)$ has variance $(\sigma^2(\pi) + \sigma^2(\pi'))/m_0$, which we estimate with the sample variances. So the test statistic
#
# $$
# T = \frac{y(\pi') - y(\pi)}{\sqrt{s^2(\pi) + s^2(\pi')}/\sqrt{m_0}}
# $$
#
# is approximately standard normal if the two expectations are equal, and a large value of $T$ is evidence for $H_1$. Let $\Phi$ be the cumulative distribution function of the standard normal distribution, and $\Phi^{-1}$ its inverse: $\Phi^{-1}(q)$ is the number $z$ with $P(Z \le z) = q$ for a standard normal $Z$. In Python, $\Phi^{-1}(q)$ is `stats.norm.ppf(q)`. A one-sided test at significance level $\alpha$ rejects $H_0(\pi, \pi')$ if $T > \Phi^{-1}(1 - \alpha)$, that is, if
#
# $$
# y(\pi) < y(\pi') - \Phi^{-1}(1 - \alpha)\, \frac{\sqrt{s^2(\pi) + s^2(\pi')}}{\sqrt{m_0}}.
# $$
#
# For $\alpha = 0.05$, $\Phi^{-1}(0.95) = 1.64$. (Strictly, the $t$-distribution with $m_0 - 1$ degrees of freedom applies, but for $m_0 \ge 30$ the difference is small.)
#
# (sidak)=
# ### The Multiple-Testing Problem and the Šidák Correction
#
# With $k$ solutions, we discard $\pi$ as soon as one of the $k - 1$ tests against the other solutions rejects $H_0(\pi, \pi')$. Even if $\pi$ is the best solution, each of these tests can reject by chance, and the more tests, the larger the probability that at least one of them does. Compare it with a multiple-choice exam that you take together with $k - 1$ monkeys that answer at random: the more monkeys, the larger the chance that one of them beats you by luck. So if every test is done at level $\alpha = 0.05$, the probability of discarding the best solution is much larger than 5%.
#
# We want the probability of discarding the best solution to be at most $\alpha$. Therefore we do each test at a stricter, "conservative" level $\alpha^*$. If $\pi$ is the best solution, all $k - 1$ hypotheses $H_0(\pi, \pi')$ are true, and each test does not reject with probability (at least) $1 - \alpha^*$. The $k - 1$ tests are not independent: they all use the same average $y(\pi)$, so if $y(\pi)$ happens to be low, $\pi$ looks bad in all comparisons at once. This dependence is positive (the tests tend to reject together), and as a result the probability that none of the $k - 1$ tests rejects is at least what it would be for independent tests:
#
# $$
# P(\text{the best solution is not discarded}) \ge (1 - \alpha^*)^{k-1}.
# $$
#
# The **Šidák correction** chooses $\alpha^*$ such that this lower bound equals $1 - \alpha$:
#
# $$
# (1 - \alpha^*)^{k-1} = 1 - \alpha \quad\Longrightarrow\quad \alpha^* = 1 - \sqrt[k-1]{1-\alpha}.
# $$
#
# Then the best solution survives with probability at least $1 - \alpha$. For the newsvendor, with $k = 21$ and $\alpha = 0.05$, this gives $\alpha^* \approx 0.0026$ and $\Phi^{-1}(1 - \alpha^*) \approx 2.80$ instead of 1.64: a much stricter bar for declaring a solution worse, because it is tested against many other solutions at once. With $1 - \alpha^* = \sqrt[k-1]{1-\alpha}$, the set of solutions that survive the first round is
#
# $$
# I = \left\{ \pi \in S \;\middle|\; y(\pi) \ge y(\pi') - \Phi^{-1}\!\left(\sqrt[k-1]{1-\alpha}\right) \frac{\sqrt{s^2(\pi)+s^2(\pi')}}{\sqrt{m_0}} \text{ for all } \pi' \ne \pi \right\}.
# $$

# %% [markdown]
# (option-2-example)=
# ### Option 2 for the Newsvendor
#
# We use the same budget $m = 4200$ as for option 1, with $m_0 = 100$ runs per order size in the first round. That leaves $4200 - 21 \times 100 = 2100$ runs for the second round. Steps 1 to 3:

# %%
rng = np.random.default_rng(3)
budget = 4200
m0 = 100
k = len(orders)
alpha = 0.05
alpha_star = 1 - (1 - alpha) ** (1 / (k - 1))
z = stats.norm.ppf(1 - alpha_star)
print(f"alpha* = {alpha_star:.4f}, critical value = {z:.2f}")

# step 1: simulate every order size m0 times
first_round = {order: simulate(order, m0, rng) for order in orders}
y = {order: runs.mean() for order, runs in first_round.items()}
s2 = {order: runs.var(ddof=1) for order, runs in first_round.items()}

# steps 2 and 3: discard every order size that is worse than another one
survivors = []
for order in orders:
    discard = False
    for other in orders:
        if other != order:
            margin = z * np.sqrt(s2[order] + s2[other]) / np.sqrt(m0)
            if y[order] < y[other] - margin:
                discard = True
    if not discard:
        survivors.append(order)
print("I =", [int(order) for order in survivors])

# %% tags=["remove-cell"] label="option2-cis"
first_half = {o: 1.96 * np.sqrt(s2[o] / m0) for o in orders}
option2_fig = go.Figure()
for kept, color, name in [
    (True, "#1f77b4", "kept (in I)"),
    (False, "grey", "discarded"),
]:
    shown = [o for o in orders if (o in survivors) == kept]
    add_cis(
        option2_fig,
        shown,
        [y[o] for o in shown],
        [first_half[o] for o in shown],
        color,
        name,
    )
finish_ci_plot(option2_fig)

# %% [markdown]
# :::{figure} #option2-cis
# :label: fig-option2-cis
#
# 95% CIs for the expected profit of every order size after the first round of option 2 (100 runs each). The grey order sizes are discarded.
# :::
#
# [](#fig-option2-cis) shows which order sizes survive. Small order sizes are discarded even though their CIs are narrow: they are clearly worse than the best order sizes. In step 4 we divide the remaining 2100 runs over the survivors, add them to the runs of the first round, and select the order size with the highest average:

# %%
n_each = (budget - k * m0) // len(survivors)
final = []
for order in survivors:
    extra_runs = simulate(order, n_each, rng)
    all_runs = np.concatenate([first_round[order], extra_runs])
    final.append(all_runs.mean())
best = int(np.argmax(final))
print(f"{n_each} extra runs per surviving order size")
print(f"selected order size: {survivors[best]}, estimate: {final[best]:.3f}")

# %% [markdown]
# The five survivors now get 520 runs each instead of 200. To compare the two options fairly, we repeat both 1000 times with the same budget, with option 2 wrapped in a function `option_2` that follows the code above line by line:


# %% tags=["hide-input"]
def option_2(budget, m0, alpha, rng):
    """Discard bad order sizes first: return the selected order size."""
    k = len(orders)
    alpha_star = 1 - (1 - alpha) ** (1 / (k - 1))
    z = stats.norm.ppf(1 - alpha_star)
    first_round = {order: simulate(order, m0, rng) for order in orders}
    y = {order: runs.mean() for order, runs in first_round.items()}
    s2 = {order: runs.var(ddof=1) for order, runs in first_round.items()}
    survivors = []
    for order in orders:
        discard = False
        for other in orders:
            if other != order:
                margin = z * np.sqrt(s2[order] + s2[other]) / np.sqrt(m0)
                if y[order] < y[other] - margin:
                    discard = True
        if not discard:
            survivors.append(order)
    n_each = (budget - k * m0) // len(survivors)
    final = []
    for order in survivors:
        extra_runs = simulate(order, n_each, rng)
        all_runs = np.concatenate([first_round[order], extra_runs])
        final.append(all_runs.mean())
    return survivors[int(np.argmax(final))]


# %%
rng = np.random.default_rng(4)
selected_2 = np.array([option_2(budget, m0, alpha, rng) for _ in range(1000)])
print(f"option 1: best order size selected {np.mean(selected_1 == 8):.2f}")
print(f"option 2: best order size selected {np.mean(selected_2 == 8):.2f}")

# %% [markdown]
# With the same budget, option 2 finds the best order size more often than option 1, because it spends more runs where it matters: on the order sizes that are hard to tell apart. It does not remove the problems of noise completely: the expected profits of order sizes 7 and 9 are only €0.03 and €0.08 below that of order size 8, and telling them apart reliably needs many more runs.
#
# :::{exercise}
# :label: ex-8-2
#
# Consider a newsvendor problem with demand Poisson distributed with average 15, $p=1$, $c=0.75$, and $\pi \in S = \{1,2,\dots,50\}$.
#
# a. Simulate the profit 1M times for each value of $\pi$ and determine the optimal $\pi$ and its value.
#
# b. Now we only can do 5000 simulations in total. Split these equally over $S$ and determine the highest value and the corresponding $\pi$. Repeat this a number of times and observe what happens.
# :::
#
# :::{exercise}
# :label: ex-8-3
#
# Apply option 2 of ranking and selection to the situation of the exercise above with a budget of 5000 simulations. Which parts of the code above do you have to change? How many order sizes survive the first round?
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §8.2.
# - Šidák, Z. (1967). "Rectangular confidence regions for the means of multivariate normal distributions." *Journal of the American Statistical Association*, 62:626–633.
