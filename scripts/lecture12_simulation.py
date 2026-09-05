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
# Simulation can be used in cases where we know exactly how the attributes or components of some system interact to give a dependent value or output. Therefore simulation is a purely predictive method by which we can model any form of dependency. However, contrary to machine-learning models, the input is random. Simulation determines the random impact of the input on the output. For example, we can quantify the impact on waiting times of having an additional cashier in a supermarket or an additional lane in a road network. In these examples, the randomness comes from the unknown behavior of the customers or drivers.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe Monte Carlo and discrete-event simulation
# - perform Monte Carlo simulations in Python
# - translate business simulation problems into simulation models
# - reflect on the usefulness of simulation in practice

# %% [markdown]
# ## Monte Carlo Simulation
#
# Suppose there are $k$ inputs and some known function $g$ that relates the inputs to the output, i.e., for inputs $x_1, \dots, x_k$ the output is $y = g(x_1, \dots, x_k)$. The central question in simulation is: can we give a reliable estimate of the expected output $EY = Eg(X_1, \dots, X_k)$? We assume that $X_1, \dots, X_k$ are know random variables which we often assume to be independent. We estimate $EY$ by repeatedly sampling from $(X_1, \dots, X_k)$ and computing $g$. Let us call the outcomes $Y_1, \dots, Y_n$. Then, according to the [law of large numbers](lecture12_variability-recap.ipynb#lln), the average $\bar Y$ is an estimator for $EY$.
#
# :::{note} Example: Project Duration
# :label: eg-5-1
#
# A project consists of a number of activities with precedence constraints: an activity can only be started when all the preceding ones are finished. For example, the roof of a house can only be constructed when the walls are finished. Clearly, project planning is an important part of project management.
#
# The essence of projects is that each project is different. Therefore we cannot predict activity durations with certainty (as we can in manufacturing, for example following the lean approach). As a simple example, suppose we have two parallel activities followed by a third activity. Then the duration of the project is $g(x_1,x_2,x_3) = \max\{x_1,x_2\} + x_3$. Assume all three durations have uniform $[0,2]$ distributions (which have mean equal to 1).
# :::
#
# We can approximate the expected duration of the project as follows:

# %%
import numpy as np

rng = np.random.default_rng(0)
n = 10000
durations = np.maximum(rng.uniform(0, 2, n), rng.uniform(0, 2, n)) + rng.uniform(0, 2, n)
print("estimated expected duration:", durations.mean())

# %% [markdown]
# Note that this is substantially more than 2, which would have been the answer in the case of deterministic durations with the same mean. Mistaking the duration of the means for the mean of the durations is a common mistake, called the strong form of the flaw of averages by Savage (2012).
#
# It is logical to question the accuracy of the answer. In simulation it is common to construct a [confidence interval](lecture12_variability-recap.ipynb#confidence-intervals) for the outcome. As discussed there, a 95% CI is given by $[m - 2s/\sqrt n, m + 2s/\sqrt n]$, with $m$ the average outcome and $s$ its SD.

# %%
m, s = durations.mean(), durations.std(ddof=1)
print("sample SD:", s)
print("95% CI:", (m - 2 * s / np.sqrt(n), m + 2 * s / np.sqrt(n)))

# %% [markdown]
# Note that, as long as the simulations are relatively simple, we can take the sample size $n$ as big as we like and thereby obtain an arbitrarily accurate answer.
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
# :::{note} Simulating Without a Programming Language
# Spreadsheets such as Excel are less appropriate for simulation. Recalculating the sheet resamples all random variables, but does not directly give a CI: that requires putting the whole simulation on one row and copying that row many times, or a dedicated add-in (such as Crystal Ball). This is one of several reasons this course simulates in Python instead: `numpy` samples thousands of realizations at once (as in the code above), and a CI is just two more lines of code away.
# :::
#
# :::{note} Tail Probabilities
# Sometimes we are not interested in the expectation of $g(X_1, \dots, X_k)$, but in probabilities of the form $P(g(X_1,\dots,X_k) \ge \alpha)$. However, a probability can be written as an expectation:
#
# $$
# P(g(X) \ge \alpha) = \sum_{g(x) \ge \alpha} P(X=x) = \sum I\{g(x) \ge \alpha\} P(X=x) = EI(g(X) \ge \alpha),
# $$
#
# with $I$ a special function, which is 1 when the argument is true and 0 otherwise, called the indicator function. Thus we end up estimating the expectation of the 0/1 function $I(g(X) \ge \alpha)$.
# :::
#
# For example, in the project planning example we are interested in the fraction of times that the project takes more than 2 time units:

# %%
above_2 = durations > 2
m, s = above_2.mean(), above_2.std(ddof=1)
print("fraction above 2:", m, "95% CI:", (m - 2 * s / np.sqrt(n), m + 2 * s / np.sqrt(n)))
#
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
# Sometimes a system is too complex to be modeled by some function $g$. This is especially the case when it concerns a process that evolves randomly over time. In such a situation discrete-event simulation (DES) is required. A central concept in DES is the state, which changes at random points in time. The state is discrete in nature, hence the name. The output or performance is usually a function of the state, averaged over time. The simulation is often terminated after a fixed amount of time, or when some condition is satisfied. We are usually interested in the expectation of the output measure. As the output of every run is a random variable, we can calculate a CI in the same way as we did for the Monte Carlo simulation.
#
# :::{note} Example: Service Centers and Warehouses
# :label: eg-5-3
#
# The evolution of waiting queues in a service center or the inventory positions in a warehouse are typical examples of processes that can be modeled using DES. Examples in other areas are the evolution of a disease in a body or the state of a communication network.
#
# In these examples, the state refers to the number of customers in the service center, the stock in the warehouse, the extent to which the disease has progressed, or the numbers of data packets in each buffer in the network routers, respectively.
#
# In the service center we might be interested in the average waiting time during a day. In the warehouse we might be interested in the long-run probability of having no stock. In the case of the progression of a disease we might be interested in the time until death, while in the network situation we might be interested in the percentage of packets lost due to buffer overflow.
# :::
#
# When employing DES there are two very distinct options: you can program the simulation in a programming language, or you can use a (graphical) tool. For certain programming languages there are libraries available with useful entities for simulation, but most code has to be programmed. Graphical simulation tools require less programming. For an impression of how such a tool works see [](#fig-arena-des). You can drag and drop components at the left to make a model in the middle. By clicking on the components they can be configured. A large number of graphical simulation tools exist, mostly proprietary, some of them focused on specific applications. The advantages of both methods are clear: programming offers flexibility and computational speed; using a tool offers speed in implementation plus a configurable graphical interface to impress customers.

# %% [markdown]
# :::{figure} images/lecture12_fig5.1.png
# :label: fig-arena-des
#
# An impression of a simple model in the Arena DES tool.
# :::

# %% [markdown]
# :::{note} Object-Oriented Programming
# Simulation lends itself perfectly to object-oriented (OO) programming, which is the paradigm behind many modern programming languages such as Java, C++ and C#. In fact, the simulation language Simula, developed in the 1960s, is generally considered to be the first OO programming language. It had a considerable influence on current-day OO languages.
# :::
#
# :::{exercise}
# :label: ex-5-4
#
# Consider a small intensive care units with 2 beds. Patients arrive with exponentially distributed interarrival times, on average every 8 hours. Patients stay for a lognormal duration with parameters 1 and 1. When both beds are occupied patients are transferred to a different hospital. Simulate this ICU for one week and count the number of transfers. Do this for multiple runs and construct a CI.
#
# This exercise requires programming experience. It is useful to store and update at the time of each event the current time, the number of occupied beds, and times of the next arrival and departures. There is a Wikipedia page with details of the exponential distribution.
# :::
#
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
#
# :::{note} Long-Run Performance
# Sometimes there is no natural termination moment for the simulation. In the supermarket example a day might be the right time frame, but in the network simulation there might not be such a moment. We are interested in the long-run stationary performance for constant parameters. Under certain conditions it can be shown (using the LLN) mathematically that the long-run average performance approaches the long-run expected performance. Because we cannot simulate for an infinitely long period, and because a single run does not give us information on the variability, it is customary to take the average over a number of runs. To avoid different "start-up" behavior, the first part of each simulation is not counted. The figure below illustrates this for a service center with 10 counters. We clearly see the average over 100 runs increasing from the empty initial situation to around 15, and two runs constantly fluctuating.
#
# :::{figure} images/lecture12_box5.5.png
# :label: fig-long-run-performance
#
# Number of customers over time for a service center with 10 counters — the average over 100 runs rises from empty to around 15, while two individual runs fluctuate constantly.
# :::
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
# - Ross, S.M. (1996). *Simulation*, 6th ed. Academic Press.
# - Savage, S. (2012). *The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty*. Wiley.
