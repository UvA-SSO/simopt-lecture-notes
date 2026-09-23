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
# # Lecture 13: Simulation Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture13_simulation-optimization.ipynb)

# %%
import numpy as np

# %% [markdown]
# Up to now we considered optimization problems that involved no randomness. However, few problems in practice are completely predictable. Although sometimes replacing random variables by constants can give a decent approximation, it can also result in very wrong results, as we saw in the [project planning example](lecture8_linear-optimization.ipynb#project-planning).
#
# Elsewhere, dynamic decision problems are studied, in which multiple decisions have to be taken, and where each decision has partially unpredictable consequences for the future. In this chapter we consider once-off decisions, where the value of every solution can only be obtained through simulation. This has important implications for optimization: you are never sure if one solution is better than another, unless you simulate many times. However, this might be very time-consuming, especially when there are many possible solutions. In this chapter, we discuss methods to deal with this problem.
#
# Different names are used for this method: simulation optimization, optimization by simulation, simulation-based optimization, or simply simopt.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe the concept of simulation optimization and its various methods
# - reflect on its usefulness for solving business problems with various options and uncertain outcomes
# - solve certain simple problems using Python

# %% [markdown]
# ## Introduction
#
# A real-life problem has random parameters $X$ (demand, travel times, arrivals, ...) and a decision $\pi$ that we choose from a set of options $S$. Because we used the variable $x$ for randomness elsewhere, we switch notation here: $\pi$ is the decision, and the general problem is
#
# $$
# \max_{\pi \in S} E[r(X, \pi)],
# $$
#
# with the key extra feature that $r(X, \pi)$ can only be evaluated by *simulating* it: there is no formula we could optimize directly. This setting is called simulation optimization (also: optimization by simulation, simulation-based optimization, or simopt).
#
# :::{note} Example: Extending Earlier Simulation Examples
# :label: eg-8-1
#
# The examples from [Simulation](lecture12_simulation.ipynb) can be extended to incorporate decisions: the impact of strategic decisions of a company can be compared in the budget using simulation; the process in the emergency department of a hospital can be optimized using simulation, etc.
# :::
#
# :::{note} Example: The Newsvendor Problem
# :label: eg-8-2
#
# A common and simple stochastic optimization problem is the newsvendor problem. A newsvendor who has to buy newspapers every morning. During the day he or she sells them. Remaining newspapers at the end of the day are worthless and have to be discarded. The newsvendor has to decide how many newspapers to buy in the morning. Because demand is random, there is a risk of lost revenue and of costs for papers that are left over. What is the optimal order size if the objective is to maximize expected profit?
# :::
#
# Simulation optimization is harder than the deterministic optimization of earlier chapters in two ways that have nothing to do with the model itself: there are no optimality guarantees (a method might return a solution whose simulated performance happens to look better than the true optimum's), and each evaluation can be slow, especially for a discrete-event simulation. On top of that, we are typically restricted to a limited *simulation budget* $m$ (say, at most 10,000 runs in total); spending it wisely, rather than splitting it blindly over every option, is the central challenge of this chapter.
#
# :::{note} Random Constraints
# We could generalize the problem formulation even further, from $\max Er(\pi,X)$ with $\pi \in S$ to $\max Er(\pi,X)$ with $Eg_j(\pi,X) \le b_j$, thereby introducing random constraints. Although we will encounter a number of such problems, we will focus on the case $\pi \in S$. Problems with random constraints are very hard to solve.
# :::
#
# **Four ways to search $S$.** How we spend the simulation budget depends entirely on the size and shape of $S$:
#
# | Shape of $S$ | Strategy | Section below |
# |---|---|---|
# | $\lvert S \rvert = 2$ | comparing scenarios | [Comparing Scenarios](#comparing-scenarios) |
# | $S$ small and discrete | ranking and selection | [Ranking and Selection](#ranking-and-selection) |
# | $S$ big or countable but discrete (e.g., a grid) | local search | [Local Search](#local-search) |
# | $S$ continuous | gradient methods | [Gradient Methods](#gradient-methods) |
#
# We work through each of the four in turn, from the simplest ($|S|=2$) to the most general ($S$ continuous).

# %% [markdown]
# (comparing-scenarios)=
# ## Comparing Scenarios
#
# The simplest form of optimization is when $|S| = 2$: we just want to know whether one scenario, giving rise to random performance $X$, is better than the other, giving rise to random performance $X'$, i.e., whether $EX \neq EX'$. The most direct approach samples $X$ and $X'$ *independently* and applies the [two-independent-samples $t$-test](lecture12_variability-recap.ipynb#hypothesis-testing): perform an equal number of runs of both scenarios, then make a CI for $EX - EX'$. If this CI excludes 0 then we have evidence that one scenario is better than the other.
#
# **Matched pairs and common random numbers.** We can often do better by designing the simulation experiment more cleverly, gaining statistical power for the same budget. The key idea is to compare the two scenarios in a "fair", similar test setting rather than fully independently: take matched pairs of samples $(X_i, X_i')$ and arrange for each pair to be *positively correlated* by simulating them under similar conditions. Then $Y = X - X'$ has $EY = EX - EX'$, and instead of a two-sample test we simulate $Y_i = X_i - X_i'$ directly and build a (one-sample, [matched-pairs](lecture12_variability-recap.ipynb#hypothesis-testing)) CI for $EY$: if $0 \notin \text{CI}$, the difference is significant.
#
# This works because
#
# $$
# \sigma^2(Y) = \sigma^2(X - X') = \sigma^2(X) + \sigma^2(X') - 2\,\mathrm{Cov}(X, X'),
# $$
#
# so making $\mathrm{Cov}(X, X') > 0$ *shrinks* $\sigma^2(Y)$ compared to treating $X$ and $X'$ as independent (where the covariance term is 0): a smaller $\sigma^2(Y)$ means a narrower CI, i.e. more statistical power for the same number of runs. In practice, we induce this positive correlation with **common random numbers (CRN)**: reuse the *same* underlying random draws (e.g., the same simulated arrival times, or in general the same seed) when simulating both scenarios, instead of drawing fresh randomness for each.
#
# **Worked example: two order quantities, with and without CRN.** A shop is deciding between ordering 8 or 12 units of a product; demand is Poisson with mean 10, sold at €5 and bought at €3 per unit, with unsold units worthless. We compare the two order quantities' profit both with independent demand draws and with CRN (the same demand realizations feeding both):

# %%
rng_crn = np.random.default_rng(7)
n_crn = 5000
price_crn, cost_crn = 5, 3
order_a, order_b = 8, 12


def profit(order, demand):
    return price_crn * np.minimum(demand, order) - cost_crn * order


# common random numbers: one shared demand stream for both scenarios
demand_shared = rng_crn.poisson(10, n_crn)
diff_crn = profit(order_b, demand_shared) - profit(order_a, demand_shared)

# independent sampling: separate demand streams per scenario
diff_indep = profit(order_b, rng_crn.poisson(10, n_crn)) - profit(
    order_a, rng_crn.poisson(10, n_crn)
)

print(f"Var(diff), common random numbers: {diff_crn.var():.2f}")
print(f"Var(diff), independent sampling:  {diff_indep.var():.2f}")

# %% [markdown]
# Because both order quantities are compared against the *same* demand draws, a high-demand draw raises both scenarios' profit together, so the difference varies far less than under independent sampling, exactly the variance reduction the formula above predicts. (In the deck's own example, a five-fold reduction in the standard deviation of the difference was worth more than a 25-fold increase in the number of independent simulations: CRN can be a much cheaper way to buy precision than brute-force replication.)
#
# :::{exercise}
# :label: ex-8-1
#
# We extend the [budget exercise](lecture12_simulation.ipynb#ex-5-3) with a second product. Management has to decide between both products. All variables are again normally distributed, with mean and SD 12000 and 2000 for sales; 90 and 20 for the price; 68 and 20 for the variable costs; 165000 and 30000 for the fixed costs.
#
# a. Simulate both scenarios 10000 times and make a CI for the difference. Which one is better? Explain your answer.
#
# b. Do the same time, but using common random numbers (as discussed above). What is the difference?
# :::

# %% [markdown]
# (ranking-and-selection)=
# ## Ranking and Selection
#
# In this section we consider problems for which the number of solutions is finite and small enough that simulating *every* candidate multiple times is feasible.
#
# :::{note} Example: Newsvendor Revisited
# :label: eg-8-3
#
# A well-known problem of this type is the newsvendor problem above. Let $X$ be the random demand; $\pi$ the order size; $p$ the selling price; and $c$ the purchasing price. For given demand $x$, the profit is equal to $p \min\{x,\pi\} - c\pi$. By simulating the demand, we can simulate the profit and determine the expected profit per order size level. Possible values for $\pi$ are $0, 1, 2, \dots$ up to a certain level, for example the maximum shelf space.
# :::
#
# :::{note} Bonus: Common Random Numbers Here Too
# The [common random numbers](#comparing-scenarios) technique from the previous section is not limited to $|S|=2$: it might take a long time to get tight CIs for every candidate, especially with a complicated discrete-event simulation, and reusing the same underlying randomness (e.g. the same customer arrival moments) across candidates reduces the variability of the *differences* between them, just as it did above. The figures below illustrate this for a service process: on the left two independent simulation traces show only a small difference; on the right, the same random traces but generated with common random numbers make the difference far easier to see.
#
# :::{figure} images/lecture13_box8.2-a.png
# :label: fig-crn-independent
#
# Two independent simulation traces with a small difference in process.
# :::
#
# :::{figure} images/lecture13_box8.2-b.png
# :label: fig-crn-common
#
# The same random traces using common random numbers.
# :::
# :::
#
# :::{exercise}
# :label: ex-8-2
#
# Consider a newsvendor problem with demand Poisson distributed with average 15, $p=1$, $c=0.75$, and $\pi \in S = \{1,2,\dots,50\}$.
#
# a. Simulate the expected demand 1M times for each value of $\pi$ and determine the optimal $\pi$ and its value.
#
# b. Now we only can do 5000 simulations in total. Split these equally over $S$ and determine the highest value and the corresponding $\pi$. Repeat this a number of times and observe what happens.
# :::
#
# From this exercise, we see two major disadvantages to simopt: because of the noise we might not recognize the signal and pick the wrong solution, and because of the noise we might overestimate the value: instead of selecting the solution with the highest value we pick the one with the highest random component. To avoid this we could simulate each solution much more times, but this might take too much time. Often we are restricted to a certain number of runs: the simulation budget. As a result, we cannot avoid the disadvantages completely, but we can improve upon an equally split simulation budget.
#
# **Option 1 (equal split) versus option 2 (discard early).** Splitting the budget equally over every candidate, as in the exercise above, wastes runs on candidates that already look clearly bad after just a few simulations. A smarter option is to first simulate all solutions a limited number of times, and then discard solutions which are unlikely to be optimal, spending the rest of the budget on the survivors. Therefore, for $|S|=k$ and a simulation budget of in total $m$ runs, we first simulate each solution $m_0 < \lfloor m/k \rfloor$ times to have an initial estimation of the mean and its estimation error, by calculating the sample mean and variance of each solution. ($\lfloor \cdot \rfloor$ is the floor operator, meaning rounding down to an integer value.) Define the sample means and variances by $y(\pi)$ and $s^2(\pi)$.
#
# When can we discard a solution $\pi$? When there are only 2 solutions $\pi$ and $\pi'$, we can discard $\pi$, using a [1-sided hypothesis test](lecture12_variability-recap.ipynb#hypothesis-testing), if:
#
# $$
# y(\pi) < y(\pi') - \frac{1.64\sqrt{s^2(\pi) + s^2(\pi')}}{\sqrt{m_0}}.
# $$
#
# Note that `scipy.stats.norm.ppf(0.95)` is equal to 1.64.
#
# **The multiple-testing problem and the Šidák correction.** With $|S|=k$ candidates, discarding $\pi$ requires comparing it against *all* $k-1$ other solutions, not just one, so even the best solution can, purely by bad luck, be "beaten" by chance in at least one of these $k-1$ tests. Running each individual comparison at the standard 5% significance level would then make the *overall* chance of a false discard far higher than 5%. The Šidák correction fixes this: it chooses a stricter, "conservative" significance level $\alpha^*$ per comparison so that the *combined* probability of falsely discarding the true best solution across all $k-1$ comparisons is still the target $\alpha$ (here $\alpha=0.05$). Because the $k-1$ comparisons are (approximately) independent, requiring each to individually fail to reject at level $\alpha^*$ with probability $1-\alpha^*$ means the overall non-rejection probability is $(1-\alpha^*)^{k-1}$, so we solve
#
# $$
# (1 - \alpha^*)^{k-1} = 1 - \alpha \quad\Longrightarrow\quad \alpha^* = 1 - \sqrt[k-1]{1-\alpha}.
# $$
#
# For $|S|=k=50$ and $\alpha=0.05$ this gives $\alpha^* \approx 0.001$, i.e. `scipy.stats.norm.ppf(0.999)` $\approx 3.09$ instead of the uncorrected `scipy.stats.norm.ppf(0.95)` $\approx 1.64$: a much stricter bar for declaring a solution "beaten", exactly to compensate for testing it against many rivals at once.
#
# This leads to the following set $I$ of candidate solutions that survive the discard round:
#
# $$
# I = \left\{ \pi \;\middle|\; y(\pi) > y(\pi') - \Phi^{-1}\!\left(\sqrt[k-1]{1-\alpha}\right) \frac{\sqrt{s^2(\pi)+s^2(\pi')}}{\sqrt{m_0}},\ \pi' \ne \pi \right\}.
# $$
#
# The remaining budget $m - m_0$ is then split evenly between the candidates in $I$, and the best of those is returned.
#
# :::{exercise}
# :label: ex-8-3
#
# Apply ranking and selection to the situation of the exercise above with a budget of 5000 simulations.
# :::

# %% [markdown]
# (local-search)=
# ## Local Search
#
# If $S$ is very large or even countable (e.g., $\{1,2,\dots\}$) then we cannot start by simulating all $x \in S$ a number of times. In this situation there is often some structure that we can exploit. Just as in the case of [deterministic local search](lecture11_algorithms-heuristics.ipynb#local-search-heuristic) we define a neighborhood $N(x)$ for every $x \in S$. During each iteration we randomly choose a point in the neighborhood of the current point and simulate both solutions once. Then we move to the best of the two and we iterate again. In more detail the algorithm is as follows:
#
# **Algorithm for local search simulation optimization**
#
# ```
# 0. Set n(π) = 0 for all π ∈ S, choose π*
# 1. Select randomly π' ∈ N(π*) and simulate π* and π'
# 2. update averages y(π*) and y(π'), increase by 1 n(π*) and n(π')
# 3. If y(π*) < y(π') then π' = π*
# 4. Repeat from 1 until simulation budget is exhausted
# 5. π* = arg max_π {n(π)}
# ```
#
# This is one of the simplest algorithms that exists. Many more elaborate algorithms are described in the scientific literature. Experience shows that the quality of the final solution is improved if the algorithm is repeated with different initial solutions.
#
# **Contrast with deterministic local search.** Deterministic local search moves to $\pi'$ whenever $g(\pi') > g(\pi^*)$ and stops once no neighbor improves on the current point: a strictly greedy walk uphill. Here, by contrast, we move to $\pi'$ whenever its *estimated* average $y(\pi')$ beats $y(\pi^*)$, and that estimate is noisy: an inferior solution can look better than $\pi^*$ purely by chance, especially early on with few visits. This randomness is not simply a flaw to be tolerated: it is what lets stochastic local search escape a local optimum that a deterministic walk would get stuck in forever, at the cost of sometimes wandering to a worse point. Reporting $\arg\max_\pi n(\pi)$ (the most-visited solution) rather than $\arg\max_\pi y(\pi)$ (the best-looking one) hedges against exactly this noise: a solution visited often has had many chances to be kicked out by a better neighbor and survived, while a solution with a lucky one-off high estimate but few visits is not trusted.
#
# :::{note} Example: Shift Scheduling with Interactions
# :label: eg-8-4
#
# [Shift scheduling](lecture9_covering.ipynb#shift-scheduling) was discussed earlier. There, the required capacity per interval was given. Often, the capacity in one interval has consequences on another, and simulation is the only tool to evaluate these effects. In such a situation, simopt can be used.
# :::
#
# :::{exercise}
# :label: ex-8-4
#
# Solve the ranking-and-selection exercise above using local search with a budget of 5000. Take $N(x) = \{x-1, x+1\}$ (unless $x=1$ or 50, then $N$ is 2 or 49). Make a plot of the current solution as the algorithm progresses.
# :::

# %% [markdown]
# (gradient-methods)=
# ## Gradient Methods
#
# When $S$ is continuous (e.g., an interval, such as choosing how many seconds a traffic light stays red), the discrete neighborhoods of local search no longer apply. Instead we take continuous steps toward higher expected performance, following an estimated gradient: the direction of steepest increase of $E[r(X,\pi)]$:
#
# $$
# \pi_{k+1} = \pi_k + \gamma_{k+1} \nabla E[r(X, \pi_k)],
# $$
#
# where $\gamma_k$ is a step size. Theory recommends $\gamma_k = 1/k$ (shrinking steps guarantee convergence under fairly general conditions), but in practice a fixed or problem-tuned step size is often used instead, since $1/k$ can shrink to nearly nothing long before a good solution is reached.
#
# The gradient itself is not directly observable (as with everything else in this chapter, $E[r(X,\pi)]$ can only be simulated), so it must be *estimated*, typically via finite differences: simulate at $\pi_k + \epsilon$ and $\pi_k - \epsilon$ for a small $\epsilon$ and estimate the slope from the two averages. Because this is itself a comparison of two nearby scenarios, the same [common random numbers](#comparing-scenarios) trick applies: evaluating both perturbations with the *same* underlying randomness gives a far less noisy gradient estimate than evaluating them independently would.


# %%
def profit_at_price(price, u):
    demand = np.maximum(0.0, 60 - 2 * price + 10 * (u - 0.5))
    return price * demand


rng_grad = np.random.default_rng(11)
price_k, eps, gamma0 = 10.0, 0.5, 4.0
for k in range(1, 6):
    u_shared = rng_grad.uniform(0, 1, 2000)  # CRN across the two perturbed evaluations
    grad_hat = (
        profit_at_price(price_k + eps, u_shared).mean()
        - profit_at_price(price_k - eps, u_shared).mean()
    ) / (2 * eps)
    price_k += (gamma0 / k) * grad_hat
    print(f"step {k}: price = {price_k:.2f}, estimated gradient = {grad_hat:.2f}")

# %% [markdown]
# ## Additional Reading
#
# There are few accessible books on simopt. Nelson (2013) is a textbook on simulation that includes a chapter on simopt; Fu (2002) is an accessible introduction to the subject.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 8, "Simulation Optimization."
# - Fu, M.C. (2002). "Optimization for simulation: Theory vs. practice." *INFORMS Journal on Computing*, 14:192–215.
# - Šidák, Z. (1967). "Rectangular confidence regions for the means of multivariate normal distributions." *Journal of the American Statistical Association*, 62:626–633.
# - Nelson, B.L. (2013). *Foundations and Methods of Stochastic Simulation*. Springer.
