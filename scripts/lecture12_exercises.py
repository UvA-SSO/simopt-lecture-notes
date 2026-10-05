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
# description: "Homework exercises for Lecture 12 on variability and simulation."
# thumbnail: null
# ---
# # Lecture 12: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture12_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in the other Lecture 12 notebooks check what you just
# read. This notebook collects the larger exercises for Lecture 12: independent problems
# worth more time. Exercises 4 and 6 ask you to read Python code; you do not need to
# write a simulation from scratch.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::

# %% [markdown]
# ## Homework Exercises

# %% [markdown]
# :::{exercise}
# :label: hw-12-1
#
# In a certain board game you roll 2 dice and take the maximum. You decide to simulate
# this.
#
# a. You run 1000 simulations. What type of distribution do you expect
#    (continuous/discrete, possible values, a known distribution, and if so, why)?
#
# b. You run 100 simulations and take the average, and repeat this 1000 times. What type
#    of distribution do you expect for these averages (continuous/discrete, possible
#    values, a known distribution, and if so, why)?
#
# c. You run 2500 simulations. The average is 4.5 and the standard deviation is 1.4. Give
#    a 95% confidence interval for the expected outcome and explain its meaning.
#
# d. Compute the expected outcome using distributions (a formula in numbers suffices, no
#    need to work out every fraction).
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-1
# :label: sol-hw-12-1
# :class: dropdown
#
# a. A discrete distribution, with possible outcomes $1, 2, 3, 4, 5, 6$.
#
# b. Still a discrete distribution in principle (there are only finitely many possible
#    averages of 100 die-max outcomes), but by the central limit theorem it will look
#    close to a normal (continuous) distribution.
#
# c. The confidence interval is
#    $\left[4.5 - \frac{2 \times 1.4}{\sqrt{2500}}, 4.5 + \frac{2 \times 1.4}{\sqrt{2500}}\right]
#    = [4.444, 4.556]$. We are 95% confident that the true expected outcome lies in this
#    interval. More precisely: if we repeated this experiment many times and built a
#    95%-CI each time in the same way, about 95% of those intervals would contain the
#    true expected value. Any single realized interval either does or does not contain
#    it; no probabilistic statement applies to it on its own.
#
# d. All 36 equally likely outcomes of (die 1, die 2) and their maximum:
#
#    | die 2 \\ die 1 | 1 | 2 | 3 | 4 | 5 | 6 |
#    |---|---|---|---|---|---|---|
#    | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
#    | 5 | 5 | 5 | 5 | 5 | 5 | 6 |
#    | 4 | 4 | 4 | 4 | 4 | 5 | 6 |
#    | 3 | 3 | 3 | 3 | 4 | 5 | 6 |
#    | 2 | 2 | 2 | 3 | 4 | 5 | 6 |
#    | 1 | 1 | 2 | 3 | 4 | 5 | 6 |
#
#    Each outcome has probability $1/36$, and the value $k$ appears $2k - 1$ times
#    (1, 3, 5, 7, 9, 11 times for $k = 1, \dots, 6$), so
#
#    $$
#    E[\max] = \frac{1 \times 1 + 3 \times 2 + 5 \times 3 + 7 \times 4 + 9 \times 5 + 11 \times 6}{36}
#    = \frac{161}{36} \approx 4.472.
#    $$
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-12-2
#
# Consider a random variable $X$ with $\Pr(X = 0) = 1/4$, $\Pr(X = 2) = 1/2$,
# $\Pr(X = 6) = 1/8$, $\Pr(X = 10) = 1/8$.
#
# a. Is $X$ discrete or continuous?
#
# b. Determine $E[X]$.
#
# c. Determine $E[aX + b]$ for constants $a$ and $b$.
#
# d. Determine $\text{Var}[X]$.
#
# e. Describe how you can simulate samples from $X$.
#
# f. Using samples $u_1 = 0.11$, $u_2 = 0.98$, $u_3 = 0.47$ from a $\text{Uniform}[0, 1]$
#    distribution, sample three realizations of $g(X) = X^2$ and give an approximation of
#    $E[g(X)]$ based on these samples.
#
# g. Discuss the limitation of the approximation for $E[g(X)]$ from part f.
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-2
# :label: sol-hw-12-2
# :class: dropdown
#
# a. $X$ takes countably many values, so it is discrete.
#
# b. $E[X] = 0 \cdot \tfrac14 + 2 \cdot \tfrac12 + 6 \cdot \tfrac18 + 10 \cdot \tfrac18 =
#    \tfrac{24}{8} = 3$.
#
# c. Using $E[X + Y] = E[X] + E[Y]$, $E[b] = b$ for a constant $b$, and $E[aX] = aE[X]$
#    for a constant $a$:
#
#    $$
#    E[aX + b] = E[aX] + E[b] = aE[X] + b = 3a + b,
#    $$
#
#    using part b for the last step.
#
# d. $\text{Var}[X] = E[(X - E[X])^2] = (0-3)^2 \tfrac14 + (2-3)^2 \tfrac12 + (6-3)^2
#    \tfrac18 + (10-3)^2 \tfrac18 = 2.25 + 0.5 + 1.125 + 6.125 = 10$.
#
# e. Split $[0, 1]$ into consecutive intervals with widths equal to the probabilities:
#    $[0, \tfrac14)$ for $X = 0$, $[\tfrac14, \tfrac34)$ for $X = 2$,
#    $[\tfrac34, \tfrac78)$ for $X = 6$, $[\tfrac78, 1]$ for $X = 10$. Draw
#    $u \sim \text{Uniform}[0, 1]$ and return the $X$-value whose interval contains $u$.
#
# f. Using the intervals from part e: $u_1 = 0.11$ falls in $[0, \tfrac14)$, so
#    $X_1 = 0$; $u_2 = 0.98$ falls in $[\tfrac78, 1]$, so $X_2 = 10$; $u_3 = 0.47$ falls in
#    $[\tfrac14, \tfrac34)$, so $X_3 = 2$. Then $g(X_1), g(X_2), g(X_3) = 0, 100, 4$, and
#    the sample average is $(0 + 100 + 4)/3 = 104/3 \approx 34.7$.
#
# g. Only 3 samples is far too few for the law of large numbers to give a reliable
#    approximation of $E[g(X)]$; the estimate in part f should be treated with strong
#    suspicion.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-12-3
#
# A retailer at a market sells bananas: selling price 0.6 euro, buying cost 0.2 euro, and
# any leftovers can be sold to a local farmer afterward for 0.1 euro each. Demand follows
# a Poisson distribution with mean 50. The retailer decides how many bananas $\pi$ to buy.
#
# a. Give a function $r(X, \pi)$ for the profit in terms of demand $X$ and order size
#    $\pi$.
#
# b. Simulation produced demand samples 50, 45, 39. Calculate the sample average and
#    variance of the profit for $\pi = 45$ based on these samples.
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-3
# :label: sol-hw-12-3
# :class: dropdown
#
# a. $r(X, \pi) = 0.6 \min(X, \pi) - 0.2\pi + 0.1 \max(\pi - X, 0)$: revenue from what is
#    sold, minus the purchase cost of everything ordered, plus salvage revenue on
#    leftovers ($\pi - X$ leftover units when $\pi \ge X$, 0 otherwise).
#
# b. For $\pi = 45$: demand 50 gives profit $0.6 \times 45 - 0.2 \times 45 - 0.1 \times 0 =
#    18$ euro; demand 45 gives $0.6 \times 45 - 0.2 \times 45 - 0.1 \times 0 = 18$ euro;
#    demand 39 gives $0.6 \times 39 - 0.2 \times 45 + 0.1 \times 6 = 15$ euro. The sample
#    average is $(18 + 18 + 15)/3 = 17$ euro, and the sample variance (dividing by $n-1$)
#    is $s^2 = \big((18-17)^2 + (18-17)^2 + (15-17)^2\big) / 2 = 6/2 = 3$.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-12-4
#
# A project consists of the activities A, B and C. A and B can be done at the same
# time, and C starts when both are finished. The following code simulates the finish
# time of the project:
#
# ```python
# import numpy as np
#
# rng = np.random.default_rng()
# n_runs = 4
# a = rng.uniform(1, 5, n_runs)
# b = rng.uniform(1, 5, n_runs)
# c = rng.uniform(1, 3, n_runs)
#
# finish = np.maximum(a, b) + c
# print(finish)
# print(finish.mean())
# print((finish > 5).mean())
# print(np.max(finish))
# ```
#
# Suppose that in one run of this code, the three sampling lines give (rounded to one
# decimal) `a` = [2.0, 4.5, 1.5, 3.0], `b` = [3.5, 2.0, 2.5, 4.0] and
# `c` = [1.0, 2.5, 1.5, 2.0]. The `numpy` functions used here are explained in the
# table in [Simulating with numpy Arrays](lecture12_monte-carlo.ipynb#numpy-arrays).
#
# a. What are the distributions of the durations of A, B and C, and their expected
#    durations?
#
# b. What do the four `print` statements print? Explain the difference between
#    `np.maximum` and `np.max`.
#
# c. Compute the finish time for the expected durations. Do you expect the true
#    expected finish time to be smaller than, equal to or larger than this number?
#    Explain.
#
# d. Give a 95% confidence interval for the expected finish time based on these four
#    runs. How would you change the code to get a confidence interval that is about ten
#    times narrower?
#
# e. Write one line of code that estimates the probability that activity A takes
#    longer than activity B. What does it give for the four runs above?
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-4
# :label: sol-hw-12-4
# :class: dropdown
#
# a. A and B are uniform on $[1, 5]$, with expected duration 3; C is uniform on
#    $[1, 3]$, with expected duration 2.
#
# b. `np.maximum(a, b)` compares the two arrays run by run, giving
#    [3.5, 4.5, 2.5, 4.0], so `finish` is [4.5, 7.0, 4.0, 6.0]. The four statements
#    print:
#
#    - the array [4.5, 7.0, 4.0, 6.0];
#    - the average $(4.5 + 7.0 + 4.0 + 6.0)/4 = 5.375$;
#    - `finish > 5` is [False, True, False, True], and its mean is the fraction of
#      `True`, 0.5;
#    - `np.max(finish)` is the largest value in the array, 7.0.
#
#    So `np.maximum` takes the maximum of two arrays per run and returns an array,
#    while `np.max` returns one number: the largest value of a single array.
#
# c. $\max(3, 3) + 2 = 5$. The true expected finish time is larger (flaw of
#    averages): the project waits for the slower of A and B, and
#    $E[\max(X_A, X_B)] > \max(EX_A, EX_B)$. A delay of either activity delays the
#    project, while an early finish of one of them does not help if the other is late.
#    (In fact $E[\max(X_A, X_B)] = 1 + 4 \times \tfrac23 \approx 3.67$, so the expected
#    finish time is about 5.67.)
#
# d. The deviations from the mean are $-0.875, 1.625, -1.375, 0.625$, so
#    $s^2 = (0.766 + 2.641 + 1.891 + 0.391)/3 = 5.6875/3 \approx 1.90$ and
#    $s \approx 1.38$. The CI is $5.375 \pm 2 \times 1.38/\sqrt 4 = 5.375 \pm 1.38$, so
#    about $[4.00, 6.75]$. With only four runs this CI is not reliable: the normal
#    approximation needs many more runs. The width decreases with $\sqrt n$, so a ten
#    times narrower CI needs $10^2 = 100$ times as many runs: `n_runs = 400`.
#
# e. `(a > b).mean()`. For the four runs, `a > b` is [False, True, False, False], which
#    gives 0.25. (The true probability is 0.5, by symmetry.)
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-12-5
#
# Jobs arrive randomly at a one-machine shop. Time between arrivals is exponential with
# mean 2 hours; manufacturing time is uniform between 1.1 and 2 hours. The machine is
# idle at time 0.
#
# a. Determine the inter-arrival times of the first four jobs using
#    $u = 0.64, 0.80, 0.34, 0.78$ drawn from $\text{Uniform}[0, 1]$ (round to 2 decimals).
#
# b. Determine the manufacturing time of the first four jobs using
#    $u = 0.70, 0.99, 0.54, 0.88$ drawn from $\text{Uniform}[0, 1]$ (round to 2 decimals).
#
# c. Determine the arrival and departure times of the jobs using parts a and b, and draw
#    the number of jobs at the machine over time.
#
# d. What is the average number of jobs per hour at the machine, based on the simulation
#    from part c, up to 10 hours (round to 2 decimals)?
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-5
# :label: sol-hw-12-5
# :class: dropdown
#
# a. Inverse-CDF sampling for an exponential with mean 2 (rate $\lambda = 1/2$) is
#    $-\ln(u) \times 2$. This gives inter-arrival times $0.89, 0.45, 2.16, 0.50$ hours.
#
# b. Inverse-CDF sampling for $\text{Uniform}[1.1, 2]$ is $1.1 + 0.9u$. This gives service
#    times $1.73, 1.99, 1.59, 1.89$ hours.
#
# c. A job can only start once it has arrived *and* the machine is idle. Using parts a
#    and b:
#
#    | job | arrival time | departure time |
#    |---|---|---|
#    | 1 | 0.89 | 2.62 |
#    | 2 | 1.34 | 4.61 |
#    | 3 | 3.50 | 6.20 |
#    | 4 | 3.99 | 8.09 |
#
#    The number of jobs at the machine over time is then a step function: 0 until 0.89,
#    up to 1 when job 1 arrives, up to 2 when job 2 arrives (at 1.34) while job 1 is still
#    in service, down to 1 when job 1 departs (2.62), up to 2 when job 3 arrives (3.50),
#    up to 3 when job 4 arrives (3.99) while job 2 is still in service, down to 2 when
#    job 2 departs (4.61), down to 1 when job 3 departs (6.20), and down to 0 when job 4
#    departs (8.09).
#
# d. Integrating the step function from part c and dividing by the 10-hour horizon:
#
#    $$
#    \frac{1(1.34-0.89) + 2(2.62-1.34) + 1(3.50-2.62) + 2(3.99-3.50) + 3(4.61-3.99) + 2(6.20-4.61) + 1(8.09-6.20)}{10} = \frac{11.8}{10} = 1.18.
#    $$
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-12-6
#
# A shop sells a product and uses an $(s, S)$ inventory policy: as soon as the stock
# drops to the reorder point $s$ or below, it orders enough to bring the stock back up to
# $S$. The order is delivered after a fixed lead time. Customers arrive with
# exponential times between arrivals and each want 1 to 5 units. A customer who wants
# more than the stock gets the remaining stock, and this is counted as a stockout. The
# following discrete-event simulation counts the stockouts:
#
# ```python
# import numpy as np
#
# reorder_point, order_up_to, lead_time = 3, 8, 1.0
# mean_interarrival = 0.5
# horizon = 3.0
#
# rng = np.random.default_rng()
# stock = order_up_to
# time = 0.0
# next_demand = rng.exponential(mean_interarrival)
# next_delivery = np.inf
# stockouts = 0
# demands = 0
#
# while time < horizon:
#     time = min(next_demand, next_delivery)
#     if time >= horizon:
#         break
#     if next_demand <= next_delivery:
#         demands += 1
#         demand_size = rng.integers(1, 6)
#         if demand_size > stock:
#             stockouts += 1
#             stock = 0
#         else:
#             stock -= demand_size
#         next_demand = time + rng.exponential(mean_interarrival)
#         if stock <= reorder_point and next_delivery == np.inf:
#             next_delivery = time + lead_time
#     else:
#         stock = order_up_to
#         next_delivery = np.inf
#     print(f"time {time:.1f}: stock {stock}")
#
# print(f"stockouts: {stockouts} out of {demands} demands")
# ```
#
# Some syntax: `np.inf` is infinity, `rng.integers(1, 6)` is a random integer from 1 to
# 5 (6 is excluded), `stock -= demand_size` means
# `stock = stock - demand_size`, and the `while` loop repeats its block as long as
# `time < horizon`, unless `break` ends it earlier.
#
# a. What is the state of this simulation, and what are the events? Which variables
#    hold the times of the next events?
#
# b. Why does the condition for placing an order contain
#    `next_delivery == np.inf`?
#
# c. Suppose the random generator gives the times between demands 0.4, 0.5, 0.3, 0.6,
#    0.9, 0.5 (in this order) and the demand sizes 4, 2, 3, 1, 5. What does the code
#    print?
#
# d. The shop switches to ordering a fixed quantity of 6 units each time, instead of
#    ordering up to $S$. Which line do you change, and how?
#
# e. Instead of the number of stockouts, the shop wants to know the total number of
#    units that customers wanted but did not get. Which lines do you change or add?
#
# f. The shop wants to estimate the fraction of demands that end in a stockout over a
#    year. Explain how you would use this simulation to give a 95% confidence interval
#    for it.
# :::
#

# %% [markdown]
#
# :::{solution} hw-12-6
# :label: sol-hw-12-6
# :class: dropdown
#
# a. The state is the stock (`stock`), together with whether an order is on its way.
#    There are two types of events: a demand (a customer arrives) and a delivery. Their
#    times are `next_demand` and `next_delivery`; `next_delivery` is `np.inf` when no
#    order is on its way.
#
# b. It makes sure that the shop does not place a new order while an earlier order is
#    still on its way. Without it, every demand that leaves the stock at or below the
#    reorder point would schedule a new delivery, overwriting the time of the earlier
#    one.
#
# c. Following the events:
#
#    | time | event | stock after the event | remark |
#    |---|---|---|---|
#    | 0.4 | demand of 4 | 4 | |
#    | 0.9 | demand of 2 | 2 | $2 \le 3$: order, delivered at 1.9 |
#    | 1.2 | demand of 3 | 0 | stockout 1; an order is already on its way |
#    | 1.8 | demand of 1 | 0 | stockout 2 |
#    | 1.9 | delivery | 8 | |
#    | 2.7 | demand of 5 | 3 | $3 \le 3$: order, delivered at 3.7 |
#
#    The next demand would be at $2.7 + 0.5 = 3.2 > 3$, so the loop stops. The code
#    prints:
#
#    ```text
#    time 0.4: stock 4
#    time 0.9: stock 2
#    time 1.2: stock 0
#    time 1.8: stock 0
#    time 1.9: stock 8
#    time 2.7: stock 3
#    stockouts: 2 out of 5 demands
#    ```
#
# d. In the delivery event, replace `stock = order_up_to` by `stock = stock + 6` (or
#    `stock += 6`).
#
# e. Add `lost_units = 0` before the loop, and in the stockout case add
#    `lost_units += demand_size - stock` before `stock = 0` (after it, `stock` is 0
#    and the result would be wrong). Print `lost_units` at the end.
#
# f. Set the horizon to a year and run the simulation many times (say 100), each time
#    with new random numbers, recording `stockouts / demands` per run. These fractions
#    are independent samples; their mean $m$ and sample SD $s$ give the CI
#    $m \pm 2s/\sqrt{100}$.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 5, "Simulation."
