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
# # Lecture 12: Simulation

# %% [markdown]
# Simulation lets us evaluate a model whose output depends on random inputs, when no closed-form formula for the expected output exists. Today we cover two flavors: Monte Carlo simulation, where a known function directly maps a batch of random inputs to an output, and discrete-event simulation (DES), for more complex processes whose state evolves randomly over time.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - sample from a distribution using the inverse transform method
# - perform Monte Carlo simulations in Python and construct a CI for the result
# - build a small discrete-event simulation
# - translate business simulation problems into simulation models

# %% [markdown]
# ## The Simulation Setting
#
# The general recipe behind everything in this lecture has four steps:
#
# 1. **Fit a probability distribution** to historical data for each random input parameter (Canvas has separate material on fitting distributions in Python; we take the fitted distribution as given here).
# 2. **Sample** realizations $x_1, x_2, \dots, x_n$ from that distribution, "in line with real-life data".
# 3. **Process** each sample through the model, giving output samples $r(x_1), r(x_2), \dots, r(x_n)$.
# 4. **Statistically analyze** the output samples, e.g. average them to approximate $E[r(X)]$.
#
# Fitting a distribution, rather than resampling historical data directly (bootstrapping), gives an unlimited stream of realistic-looking samples and avoids overfitting the simulation to one fixed, finite history: a solution that works great on the past data used to build it but not necessarily on the future it is meant to predict.
#
# :::{note} Simulating Without a Programming Language
# Spreadsheets such as Excel are less appropriate for simulation. Recalculating the sheet resamples all random variables, but does not directly give a CI: that requires putting the whole simulation on one row and copying that row many times, or a dedicated add-in. This is one of several reasons this course simulates in Python instead: `numpy` samples thousands of realizations at once, as we do below, and a CI is just two more lines of code away.
# :::

# %% [markdown]
# ## The Flaw of Averages
#
# A natural but wrong shortcut is to plug the expected value of each input directly into $r$ and call the result the expected output: $r(EX) \approx E[r(X)]$. In general,
#
# $$
# E[r(X)] \neq r(EX),
# $$
#
# an error common enough to have its own name, the *flaw of averages* (Savage, 2012).
#
# A simple case makes the direction of the error tangible: let $X$ and $Y$ be independent fair coin tosses (0 or 1), and $r(X,Y) = \max(X,Y)$. Then $EX = EY = 0.5$, so $r(EX,EY) = \max(0.5, 0.5) = 0.5$. But $r(X,Y)$ is 1 unless both tosses are 0, so $E[r(X,Y)] = P(X=1 \text{ or } Y=1) = 0.75$. Whenever $r$ involves a maximum, a minimum, or another nonlinear operation, the two quantities can differ substantially, and the gap tends to grow with the number of random inputs involved, since it becomes more likely that *at least one* of them is unusually high (or low). This is exactly the situation in, e.g., project planning: the finish time of a project is the maximum over many possible critical paths, so a few unlucky activities are enough to delay the whole project, and averaging over that risk is not the same as plugging in average durations.
#
# The rest of this lecture builds the two main tools for computing $E[r(X)]$ correctly: sampling realizations of $X$, and averaging $r$ over many of them.

# %% [markdown]
# ## Sampling a Random Variable: The Inverse Transform Method
#
# Almost every sampling technique used in practice ultimately rests on one trick, the *inverse transform method* (ITM): given a source of uniform $[0,1]$ samples $U$ (which every programming language provides), we can sample any distribution with cdf $F$ by returning $F^{-1}(U)$, where $F^{-1}(u)$ is defined as the $x$ solving $F(x) = u$ (for discrete distributions, the smallest such $x$).
#
# **Why this works.** For any $x$,
#
# $$
# P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x) = P(X \le x),
# $$
#
# using that $F$ is non-decreasing and $U$ is uniform on $[0,1]$ (so $P(U \le u) = u$). Since $F^{-1}(U)$ and $X$ have the same cdf, they have the same distribution.
#
# **Intuition.** The steeper $F$ increases on some interval, the wider that interval is when we "un-invert" it back from the $u$-axis to the $x$-axis, so a uniformly spread-out set of $u$-values lands, after applying $F^{-1}$, disproportionately often in the steep regions of $F$. Since $F$ is steep exactly where the density $f = F'$ is high, this reproduces the shape of the pdf: more samples land where the density says they should.
#
# **Worked example 1: a continuous distribution.** Suppose a component's lifetime $X$ (in years) is exponentially distributed with rate $\lambda = 0.5$ (mean $1/\lambda = 2$ years), with cdf $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$. Solving $F(x) = u$ for $x$:
#
# $$
# 1 - e^{-\lambda x} = u \quad\Longrightarrow\quad x = -\frac{\ln(1-u)}{\lambda} = F^{-1}(u).
# $$
#
# Sampling via this formula and comparing the resulting histogram to the exponential pdf:

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
rate = 0.5
u = rng.uniform(0, 1, 20000)
lifetimes = -np.log(1 - u) / rate

x_grid = np.linspace(0, 20, 200)
plt.hist(lifetimes, bins=40, density=True, alpha=0.6, label="ITM samples")
plt.plot(x_grid, stats.expon.pdf(x_grid, scale=1 / rate), label="exponential pdf")
plt.xlabel("lifetime (years)")
plt.legend()

# %% [markdown]
# (`np.random.default_rng(...).exponential(scale=1 / rate)` would do the same sampling directly, since most libraries implement common distributions this way internally, but the formula above shows exactly what is happening underneath.)
#
# **Worked example 2: a discrete distribution.** Suppose a delivery arrives 1, 2, or 3 days late with probabilities $P(X=1) = 0.2$, $P(X=2) = 0.5$, $P(X=3) = 0.3$. The cdf is a step function, $F(1) = 0.2$, $F(2) = 0.7$, $F(3) = 1$, so
#
# $$
# F^{-1}(u) = \min\{x : F(x) \ge u\} =
# \begin{cases}
# 1 & 0 \le u \le 0.2 \\
# 2 & 0.2 < u \le 0.7 \\
# 3 & 0.7 < u \le 1.
# \end{cases}
# $$
#
# In practice: partition $[0,1]$ into subintervals matching the cumulative probabilities, sample $U$, and see which subinterval it falls into.

# %%
outcomes = np.array([1, 2, 3])
cumulative_probs = np.array([0.2, 0.7, 1.0])
u2 = rng.uniform(0, 1, 20000)
delays = outcomes[np.searchsorted(cumulative_probs, u2, side="left")]

for value, prob in zip(outcomes, np.diff(np.concatenate(([0.0], cumulative_probs)))):
    print(f"P(X={value}) target {prob:.2f}, sampled {np.mean(delays == value):.3f}")

# %% [markdown]
# :::{note} Random Number Generators and Seeds
# Behind `rng.uniform` sits a pseudo-random number generator (PRNG): a deterministic formula that produces a sequence of numbers which *looks* random (passes statistical tests for randomness) but is entirely determined by a starting value, the seed. This is a feature, not a bug: `np.random.default_rng(0)` always produces the same stream, which makes simulation results reproducible. That matters for debugging and for comparing two designs under identical randomness (we return to that idea in the next lecture). True hardware randomness (e.g., based on atmospheric noise) exists but is slower and rarely necessary in practice.
# :::

# %% [markdown]
# ## Monte Carlo Simulation
#
# We can now put sampling to work. Monte Carlo simulation approximates $E[r(X)]$ as follows: generate i.i.d. samples $x_1, \dots, x_n$ of $X$ (using ITM, or a library shortcut), compute $r(x_1), \dots, r(x_n)$, and average them:
#
# $$
# \overline{r(X)} = \frac{r(x_1) + \dots + r(x_n)}{n}.
# $$
#
# **Why does this converge?** Write $Y_i = r(x_i)$, so $Y_1, \dots, Y_n$ are i.i.d. copies of the random variable $Y = r(X)$, with mean $EY$ and variance $\sigma^2(Y)$. By the [expectation and variance rules for sums](lecture12_variability-recap.ipynb#sums-of-rvs),
#
# $$
# E\left[\frac{Y_1 + \dots + Y_n}{n}\right] = \frac{\sum_i EY_i}{n} = EY, \qquad
# \sigma^2\left[\frac{Y_1 + \dots + Y_n}{n}\right] = \frac{\sum_i \sigma^2(Y_i)}{n^2} = \frac{\sigma^2(Y)}{n}.
# $$
#
# The sample average is centered exactly on $EY$, and its variance shrinks to 0 as $n \to \infty$: this is the [law of large numbers](lecture12_variability-recap.ipynb#lln), the sample average of a simulation converges to the true expected performance.
#
# :::{note} Example: Call-Center Overtime
# :label: eg-5-1
#
# A call center staffs for 45 calls per hour, but call volume $X$ fluctuates and is Poisson distributed with mean 50. Every call beyond capacity requires an overtime agent at a cost of €30. The hourly overtime cost is $r(X) = 30 \max(0, X - 45)$.
#
# Plugging in the mean, $r(EX) = 30 \max(0, 50 - 45) = 150$, understates the true expected cost: because $X$ sometimes spikes well above 50, the $\max(0, \cdot)$ kicks in harder on the bad hours than it saves on the good ones, exactly as in the flaw-of-averages example above.
# :::

# %%
n = 10000
calls = rng.poisson(50, n)
overtime_cost = 30 * np.maximum(0, calls - 45)
print("naive estimate r(EX):", 30 * max(0, 50 - 45))
print("simulated estimate E[r(X)]:", overtime_cost.mean())

# %% [markdown]
# As with any estimate based on a finite sample, we should quantify its accuracy with a [confidence interval](lecture12_variability-recap.ipynb#confidence-intervals): with sample mean $m$ and sample SD $s$, a 95% CI is $[m - 2s/\sqrt n, m + 2s/\sqrt n]$.

# %%
m, s = overtime_cost.mean(), overtime_cost.std(ddof=1)
print("sample SD:", s)
print("95% CI:", (m - 2 * s / np.sqrt(n), m + 2 * s / np.sqrt(n)))

# %% [markdown]
# The width of this CI shrinks with $1/\sqrt n$, not $1/n$: to halve it we need *four times* as many simulation runs, not twice as many. This is the price of randomness: beyond a certain point, more precision gets expensive fast.
#
# :::{exercise}
# :label: ex-5-1
#
# a. Explain the difference between the numpy functions `np.maximum` and `np.max` (used above and elsewhere respectively).
#
# b. Simulate the [project planning problem](lecture8_linear-optimization.ipynb#project-planning), with all activities having uniform distributions with mean as indicated and width $(b-a)$ equal to 2. Compute a CI.
#
# c. Simulate the project again with all activities having lognormal distributions with mean as indicated and SD 1. Note that the mean and SD of the underlying normal distributions first need to be computed.
# :::
#
# :::{note} Tail Probabilities
# Sometimes we are not interested in the expectation of $r(X)$, but in probabilities of the form $P(r(X) \ge \alpha)$. However, a probability can be written as an expectation:
#
# $$
# P(r(X) \ge \alpha) = \sum_{r(x) \ge \alpha} P(X=x) = \sum I\{r(x) \ge \alpha\} P(X=x) = E[I(r(X) \ge \alpha)],
# $$
#
# with $I$ the indicator function, 1 when its argument is true and 0 otherwise. Thus we end up estimating the expectation of the 0/1 random variable $I(r(X) \ge \alpha)$, a Monte Carlo simulation like any other.
# :::
#
# For example, in the call-center example we might ask for the fraction of hours with more than €300 of overtime cost:

# %%
above_300 = overtime_cost > 300
m, s = above_300.mean(), above_300.std(ddof=1)
print("fraction above €300:", m, "95% CI:", (m - 2 * s / np.sqrt(n), m + 2 * s / np.sqrt(n)))

# %% [markdown]
# :::{exercise}
# :label: ex-5-3
#
# The budget of most companies are set without taking the variability of the numbers into account. Consider a simple budget:
#
# $$
# \text{profit} = \text{sales} \times (\text{price} - \text{variable costs}) - \text{fixed costs}.
# $$
#
# All components depend on market situations and are random; for simplicity we assume them to be independent. They can be assumed to have normal distributions, with mean and SD 10000 and 1000 for sales, 100 and 20 for the price, 80 and 10 for the variable costs, and 100000 and 20000 for the fixed costs.
#
# a. Simulate the expected profit and determine a CI.
#
# b. Could you have calculated the answer without simulation? Explain your answer.
#
# c. Determine the probability of a loss and a CI of this probability. Hint: first read the box above on "tail probabilities".
# :::
#
# :::{note} Machine Learning versus Simulation
# Most machine-learning methods consist of two phases: first a descriptive phase in which a model is learned, within a certain class of models, such as linear with normally distributed noise. The second phase is the predictive part in which the output for new, deterministic, inputs is approximated. Simulation only consists of a predictive part, without restrictions on the model. It quantifies the consequences of uncertainty on the input to the output. Simulation can also be used for the predictive phase of a machine-learning model in case the input is random.
# :::

# %% [markdown]
# ## Discrete-Event Simulation
#
# Monte Carlo simulation needs $r$ to be a known function of a fixed, small set of inputs. Many real processes are too complex for that: their behavior unfolds through a sequence of random events over time, and what happens next depends on the current state, not on a fixed formula. Discrete-event simulation (DES) handles this by explicitly keeping track of a **state** (e.g., the stock in a warehouse, the number of customers in a queue), which changes only at discrete points in time: the events. Between events, nothing happens, so we can jump straight from one event to the next instead of simulating time continuously.
#
# :::{note} Example: Service Centers and Warehouses
# :label: eg-5-2
#
# The evolution of waiting queues in a service center or the inventory position in a warehouse are typical examples of processes modeled using DES. Other examples are the evolution of a disease in a body or the state of a communication network. In the service center, the state is the number of customers present; in the warehouse, the stock level; in the network, the number of packets queued at each router.
# :::
#
# The generic simulation loop repeats five steps until a stopping condition is met:
#
# 1. **Determine the next event**, the one with the smallest scheduled time among all pending events.
# 2. **Update the time** to that event's time.
# 3. **Update the state** according to what the event does (e.g., stock decreases on a demand, increases on a delivery).
# 4. **Schedule follow-up events** the current event triggers (e.g., a demand event schedules the next demand arrival).
# 5. **Record performance measures** of interest, then go back to step 1.
#
# **Worked example: an $(s, S)$ inventory system.** A shop reviews continuously and reorders whenever stock drops to or below a reorder point $s$; the resulting order brings stock back up to $S$ after a lead time. There are two event types: demand arrivals (exponential interarrival times, random demand size) and order arrivals (fixed lead time after an order is triggered). We track the number of demand events that find the shop out of stock.

# %%
reorder_point, order_up_to, lead_time = 10, 50, 2.0
mean_interarrival = 0.5
horizon = 2000.0

demand_rng = np.random.default_rng(1)
stock = order_up_to
time = 0.0
next_demand = demand_rng.exponential(mean_interarrival)
next_delivery = np.inf
stockouts = 0
demands = 0

while time < horizon:
    time = min(next_demand, next_delivery)
    if time >= horizon:
        break
    if next_demand <= next_delivery:
        demands += 1
        demand_size = demand_rng.integers(1, 6)
        if demand_size > stock:
            stockouts += 1
            stock = 0
        else:
            stock -= demand_size
        next_demand = time + demand_rng.exponential(mean_interarrival)
        if stock <= reorder_point and next_delivery == np.inf:
            next_delivery = time + lead_time
    else:
        stock = order_up_to
        next_delivery = np.inf

print(f"stockouts: {stockouts} out of {demands} demands ({stockouts / demands:.1%})")

# %% [markdown]
# :::{exercise}
# :label: ex-5-4
#
# Consider a small intensive care unit with 2 beds. Patients arrive with exponentially distributed interarrival times, on average every 8 hours. Patients stay for a lognormal duration with parameters 1 and 1. When both beds are occupied patients are transferred to a different hospital. Simulate this ICU for one week and count the number of transfers. Do this for multiple runs and construct a CI.
#
# This exercise requires programming experience. It is useful to store and update at the time of each event the current time, the number of occupied beds, and times of the next arrival and departures. There is a Wikipedia page with details of the exponential distribution.
# :::

# %% [markdown]
# ## DES Tooling
#
# There are two broad ways to build a DES: program it yourself, or use dedicated software.
#
# :::{note} Object-Oriented Programming
# Simulation lends itself perfectly to object-oriented (OO) programming, which is the paradigm behind many modern programming languages such as Java, C++ and C#. In fact, the simulation language Simula, developed in the 1960s, is generally considered to be the first OO programming language. It had a considerable influence on current-day OO languages.
# :::
#
# **Programming it yourself**, as we just did, needs no more than a general-purpose language plus a random number generator and, for anything bigger than our toy example, a proper event-list data structure (e.g., a priority queue) to keep track of many pending events efficiently. It offers full flexibility and computational speed, at the cost of more development effort.
#
# **Dedicated DES software** trades some of that flexibility for faster modeling and a graphical interface:
#
# - **SimQuick** ([simquick.net](http://simquick.net/)) implements DES logic inside Excel: a lightweight option requiring no separate installation, at the cost of Excel's usual simulation limitations.
# - **Arena** offers a full drag-and-drop graphical modeler; see [](#fig-arena-des) for an impression. You drag components from a palette on the left to build a model in the middle, then configure each by clicking on it.
# - **JaamSim** (Java Animation Modelling & Simulation) is an actively maintained, open-source (Apache 2.0) alternative with a drag-and-drop GUI and optional 3D animation, cross-platform (Windows/Linux/macOS), and extensible in Java for custom components. Kristiansen et al. (2022) compare it against other open-source DES tools.
#
# Graphical tools in general are quick to build a first model in and easy to present to non-technical stakeholders, but tend to be slower to run, less flexible for anything outside their built-in components, and often commercially licensed.

# %% [markdown]
# :::{figure} images/lecture12_fig5.1.png
# :label: fig-arena-des
#
# An impression of a simple model in the Arena DES tool.
# :::

# %% [markdown]
# **Validation**
#
# Statistics and ML also play an important role in DES, but in a different way than when you apply them directly. In simulation the system under study is considered to be composed of components, who interact in a known way. However, to derive the parameters of the components and their interaction we need statistics and ML. Validation is concerned with the question to which extend the simulation reflects reality, i.e., to which extent errors in the components and their interaction propagate to the level of the performance measures. Therefore, the parameters of the components are determined using appropriate ML techniques, and then the performance of the simulation is compared to that of the real system. An accurate approximation of the components does not guarantee an accurate output: (simulation) models can be more or less robust to parameter errors.
#
# Validation is rarely easy, often because of a lack of data. Quite often the differences between reality and simulation are so big that statistical tests for equality are always rejected. This does not mean that simulation is useless. In evaluating the differences, we should always take the goal of the simulation into account. Still, in many situations simulations can only be validated after considerable effort or even tuning (where certain parameters are adapted to make the simulation model fit reality). Simulation should only be used after careful consideration: implementing simulation is time-consuming and there is no guarantee of reliable results.
#
# :::{note} Example: Emergency Department
# :label: eg-5-4
#
# Consider an emergency department (ED) of a hospital. Every ED has limited care facilities, often leading to congestion and delays. Four hours is generally considered to be the limit to the length of stay of patients at an ED. Our ED wants to analyze the factors that lead to higher numbers of patients staying longer than 4 hours.
#
# Simulation requires many resources within the ED to be modeled: triage nurses, doctors with different specialties, beds, radiology equipment, etc. For all these resources, parameters such as their duration have to be determined, as well as the routing between the resources, priority of treatment, etc. Reliable data is hard to get, among other reasons because data entry is of little importance and because of the omnipresence of ad-hoc decisions. This lack of reliable data often translates into output that is far from reality. This makes simulation of limited use to systems where human decision making plays such a central role.
#
# We could use an ML approach directly. We should engineer our features as to obtain parameters of interest. Then we could train our model and determine the impact of, for example, high numbers of arrivals or lateness of doctors on the length of stay at the ED. There is no such thing as validation in ML; the risk here is overfitting.
#
# An ML approach is much faster than simulation and often gives very good results. In certain situations, we cannot avoid the use of simulations, especially when we are interested in situations where we do not have data. For example, suppose we want to change the routing in the ED. Then all components are analyzed using historical data. The model is validated on the current way of working, and then the new situation is simulated.
# :::

# %% [markdown]
# ## Long-Run Performance
#
# Sometimes there is no natural termination moment for the simulation. In the inventory example a year might be the right time frame, but in a network simulation there might not be such a moment. We are then interested in the long-run stationary performance for constant parameters. Under certain conditions it can be shown (using the LLN) mathematically that the long-run average performance approaches the long-run expected performance. Because we cannot simulate for an infinitely long period, and because a single run does not give us information on the variability, it is customary to take the average over a number of runs, exactly as for the Monte Carlo CI above. To avoid different "start-up" behavior (e.g., our inventory example starting completely full), the first part of each simulation is often excluded from the performance measure, a period known as the warm-up.
#
# The figure below illustrates this for a service center with 10 counters. We clearly see the average over 100 runs increasing from the empty initial situation to around 15, and two individual runs constantly fluctuating around it, only slowly settling down, because the state at one moment is highly correlated with the state shortly after (if the system is full now, it tends to still be full a bit later), which is why long-run simulations typically need much longer horizons than a Monte Carlo simulation needs samples to get an equally tight CI.
#
# :::{figure} images/lecture12_box5.5.png
# :label: fig-long-run-performance
#
# Number of customers over time for a service center with 10 counters: the average over 100 runs rises from empty to around 15, while two individual runs fluctuate constantly.
# :::

# %% [markdown]
# ## Additional Reading
#
# There are many books on the mathematical aspects of simulation. See, e.g., Ross (1996). Kelton, Sandowski, and Sandowski (1998) is an example of a book which is more focused on modeling and tooling (especially the discrete-event simulation tool Arena). There is a list of discrete-event software tools on Wikipedia.
#
# Savage (2012) uses simulation to explain variability and its pitfalls to layman, avoiding words such as random variable. Klastorin (2003) is an excellent book on project management.
#
# More information on the OO simulation language Simula can be found on Wikipedia.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 5, "Simulation."
# - Kelton, W.D., Sandowski, R.P., & Sandowski, D.A. (1998). *Simulation with Arena*. McGraw-Hill.
# - Klastorin, T. (2003). *Project Management: Techniques and Tradeoffs*. Wiley.
# - Kristiansen, S., Fabritius, F., & Xie, X. (2022). "A comparison of open-source discrete-event simulation software." Winter Simulation Conference.
# - Ross, S.M. (1996). *Simulation*, 6th ed. Academic Press.
# - Savage, S. (2012). *The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty*. Wiley.
