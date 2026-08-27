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
# Lecture 13: Simulation Optimization

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Simulation optimization (Ch. 8)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — Chapter 8 "Simulation Optimization" (whole)
# - **Errata applied:** none

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 8 — Simulation Optimization
#
# Up to now we considered optimization problems that involved no randomness. However, few problems in practice are completely predictable. Although sometimes replacing random variables by constants can give a decent approximation, it can also result in very wrong results, as we saw in the project planning example on page 78.
#
# In Chapter 9 we study dynamic decision problems, in which multiple decisions have to be taken, and where each decision has partially unpredictable consequences for the future. In this chapter we consider once-off decisions, where the value of every solution can only be obtained through simulation. This has important implications for optimization: you are never sure if one solution is better than another, unless you simulate many times. However, this might be very time-consuming, especially when there are many possible solutions. In this chapter, we discuss methods to deal with this problem.
#
# Different names are used for this method: simulation optimization, optimization by simulation, simulation-based optimization, or simply simopt.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe the concept of simulation optimization and its various methods
# - reflect on its usefulness for solving business problems with various options and uncertain outcomes
# - solve certain simple problems using R or Excel

# %% [markdown]
# #### 8.1 Introduction
#
# In this chapter we bring together two concepts from previous chapters: randomness and decisions. Because we used the variable $x$ for both these concepts in previous chapters, we have to change the notation: we will use $\pi$ for the decision. The problems we study in this chapter are of the form $\max E r(\pi, X)$ with $\pi \in S$ and with the additional feature that we can only approximate $Er(\pi,X)$ using simulation.
#
# **Example 8.1** The examples from Chapter 5 can be extended to incorporate decisions: the impact of strategic decisions of a company can be compared in the budget using simulation; the process in the emergency department of a hospital can be optimized using simulation, etc.
#
# **Example 8.2** A common and simple stochastic optimization problem is the newsvendor problem. A newsvendor who has to buy newspapers every morning. During the day he or she sells them. Remaining newspapers at the end of the day are worthless and have to be discarded. The newsvendor has to decide how many newspapers to buy in the morning. Because demand is random, there is a risk of lost revenue and of costs for papers that are left over. What is the optimal order size if the objective is to maximize expected profit?
#
# For a problem like the newsvendor we could simulate each solution a number of times and make confidence intervals. However, there are a number of problems with this approach. What does it mean if the CIs are non-overlapping? Or only overlapping for a small part? Furthermore, there may be so many solutions that this approach is infeasible from the beginning. And even if it is feasible, we want to concentrate on possible winners and discard solutions with really low values from the beginning. In the next sections we will discuss methods for all these issues, depending on the form of $S$.
#
# > **Box 8.1. Random constraints**
# >
# > We could generalize the problem formulation even further, from $\max Er(\pi,X)$ with $\pi \in S$ to $\max Er(\pi,X)$ with $Eg_j(\pi,X) \le b_j$, thereby introducing random constraints. Although we will encounter a number of such problems, we will focus on the case $\pi \in S$. Problems with random constraints are very hard to solve.
#
# $S$ can be of two basic forms: discrete or continuous, i.e., for example of the form $[0,1]$ or of the form $\{0,1,2,\dots\}$. We will focus on discrete problems. Note that a continuous $S$ can always be made discrete: instead of $[0,1]$ we could consider $\{0, 0.01, 0.02, \dots, 1\}$.
#
# For $S$ discrete we discuss 3 separate methods, for the following cases:
#
# - $|S| = 2$;
# - $|S|$ small;
# - $|S|$ big or even infinite.

# %% [markdown]
# #### 8.2 Comparing scenarios
#
# The simplest form of optimization is when $|S| = 2$, when we compare two scenarios to see whether the objective value of one is higher or lower than the other. In this situation we typically perform an equal number of runs of both situations. Then we make a CI of the differences. If this CI excludes 0 then we have evidence that one scenario is better than the other.
#
# **Exercise 8.1** We extend Exercise 5.3 on page 79 with a second product. Management has to decide between both products. All variables are again normally distributed, with mean and SD 12000 and 2000 for sales; 90 and 20 for the price; 68 and 20 for the variable costs; 165000 and 30000 for the fixed costs.
#
# a. Simulate both scenarios 10000 times and make a CI for the difference. Which one is better? Explain your answer.
#
# b. Do the same time, but using common random numbers (as in Box 8.2). What is the difference?

# %% [markdown]
# #### 8.3 Ranking and selection
#
# In this section we consider problems for which the number of solutions is finite and limited in size, allowing us to simulate every possible solution multiple times.
#
# **Example 8.3** A well-known problem of this type is the newsvendor problem of Example 8.2. Let $X$ be the random demand; $\pi$ the order size; $p$ the selling price; and $c$ the purchasing price. For given demand $x$, the profit is equal to $p \min\{x,\pi\} - c\pi$. By simulating the demand, we can simulate the profit and determine the expected profit per order size level. Possible values for $\pi$ are $0, 1, 2, \dots$ up to a certain level, for example the maximum shelf space.
#
# > **Box 8.2. Common random numbers**
# >
# > It might take a long computation time to get a tight CI, especially in the case of complicated discrete-event simulations. Sometimes the variability of the differences between the scenarios can be reduced by a technique called common random numbers. The idea is that for random variables that are used in both scenarios, the same outcomes are taken. As a result the runs become dependent, and often the variability is reduced. For example, in a service center where we change the service delivery process, we might take the same customer arrival moments for both scenarios. This often reduces drastically the variance and therefore also reduces the width of the CI. This method is illustrated in the figures below: the left shows two independent traces with a small difference in process; the right shows the same random traces but with common random numbers.

# %% [markdown]
# ![Box 8.2 (left): two independent simulation traces with a small difference in process](images/lecture13_box8.2-a.png)
#
# ![Box 8.2 (right): the same random traces using common random numbers](images/lecture13_box8.2-b.png)

# %% [markdown]
# **Exercise 8.2** Consider a newsvendor problem with demand Poisson distributed with average 15, $p=1$, $c=0.75$, and $\pi \in S = \{1,2,\dots,50\}$.
#
# a. Simulate the expected demand 1M times for each value of $\pi$ and determine the optimal $\pi$ and its value.
#
# b. Now we only can do 5000 simulations in total. Split these equally over $S$ and determine the highest value and the corresponding $\pi$. Repeat this a number of times and observe what happens.
#
# From this exercise, we see two major disadvantages to simopt: because of the noise we might not recognize the signal and pick the wrong solution, and because of the noise we might overestimate the value: instead of selecting the solution with the highest value we pick the one with the highest random component. To avoid this we could simulate each solution much more times, but this might take too much time. Often we are restricted to a certain number of runs: the simulation budget. As a result, we cannot avoid the disadvantages completely, but we can improve upon an equally split simulation budget.
#
# A smarter solution is to first simulate all solutions a limited number of times, and then discard solutions which are unlikely to be optimal. This selection is based on a statistical test. Therefore, for $|S|=k$ and a simulation budget of in total $m$ runs, we first simulate each solution $m_0 < \lfloor m/k \rfloor$ times to have an initial estimation of the mean and its estimation error, by calculating the sample mean and variance of each solution. ($\lfloor \cdot \rfloor$ is the floor operator, meaning rounding down to an integer value.) Define the sample means and variances by $y(\pi)$ and $s^2(\pi)$.
#
# When can we discard a solution $\pi$? When there are only 2 solutions $\pi$ and $\pi'$, we can discard $\pi$, using a 1-sided hypothesis test (see page 50), if:
#
# $$
# y(\pi) < y(\pi') - \frac{1.64\sqrt{s^2(\pi) + s^2(\pi')}}{\sqrt{m_0}}.
# $$
#
# Note that in R `qnorm(0.95)` is equal to 1.64.
#
# Now we have $|S|-1$ other solutions. The total probability of falsely discarding $\pi$ should be 0.05. For each comparison it should be $1 - \sqrt[|S|-1]{0.95}$. For $|S|=50$ this is equal to 0.001. In this case `qnorm(0.999)` gives 3.08.
#
# This leads to the following set of candidate solutions:
#
# $$
# I = \left\{ \pi \;\middle|\; y(\pi) > y(\pi') - \Phi^{-1}\!\left(\sqrt[|S|-1]{1-\alpha}\right) \frac{\sqrt{s^2(\pi)+s^2(\pi')}}{\sqrt{m_0}},\ \pi' \ne \pi \right\}.
# $$
#
# The remaining budget $m - m_0$ is equally split between the candidate solutions and the best is chosen.
#
# **Exercise 8.3** Apply ranking and selection to the situation of Exercise 8.2 with a budget of 5000 simulations.

# %% [markdown]
# #### 8.4 Local search
#
# If $S$ is very large or even countable (e.g., $\{1,2,\dots\}$) then we cannot start by simulating all $x \in S$ a number of times. In this situation there is often some structure that we can exploit. Just as in the case of deterministic local search (see page 114) we define a neighborhood $N(x)$ for every $x \in S$. During each iteration we randomly choose a point in the neighborhood of the current point and simulate both solutions once. Then we move to the best of the two and we iterate again. In more detail the algorithm is as follows:
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
# **Example 8.4** In Chapter 6 (page 99) we discussed shift scheduling. There, the required capacity per interval was given. Often, the capacity in one interval has consequences on another, and simulation is the only tool to evaluate these effects. In such a situation, simopt can be used.
#
# **Exercise 8.4** Solve Exercise 8.3 using local search with a budget of 5000. Take $N(x) = \{x-1, x+1\}$ (unless $x=1$ or 50, then $N$ is 2 or 49). Make a plot of the current solution as the algorithm progresses. You can do this in Excel or R. Doing it in R requires the use of for-loops.

# %% [markdown]
# #### 8.5 Additional reading
#
# There are few accessible books on simopt. Nelson [26] is a textbook on simulation that includes a chapter on simopt; Fu [11] is an accessible introduction to the subject.
#
# Note that we have not discussed situations in which the solution space is continuous, for example all values between 0 and 1. In these situations approximations of derivatives (in more dimensions called gradients) are very useful.
