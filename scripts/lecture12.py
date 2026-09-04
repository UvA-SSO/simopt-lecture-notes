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
# # Lecture 12: Simulation (with a Variability Recap)

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 3 — Variability *(recap)*
#
# Variability is omnipresent: without variability every moment of the day would be the same. Often there is a certain level of uncertainty about variability: we do know exactly when it is light and dark, but we do not know exactly when it will rain. Data science tries to explain, predict and control the variability we observe. The study of variability is therefore crucial for a solid understanding of data science and business analytics. This goes beyond studying the data itself: mathematical theory helps us understand the data and obtain results about the data. Mathematics and statistics impose a theoretical framework in which, under well-specified conditions, certain results are obtained. When these conditions are (approximately) verified in the data, then we can use the mathematical results, which makes us understand our data much better.
#
# In this chapter we focus on univariate data and models. We first show how to summarize data. If you have data on all items you are interested in then this might be sufficient. However, often you only have data on part of the population. Or, you want to predict future values of the data based on historical values. In this case you need to be able to distinguish between noise and signal: is the summary representative of the population or is there so much noise that a new experiment would give very different results? Do we overfit our data such that our prediction has little predictive value? To answer these types of questions, we will apply basic probability theory, especially distributions and the central limit theorem. After that, we introduce some useful hypothesis tests. This closes the circle: using results from probability theory we draw conclusions from data.
#
# This chapter is more mathematical in nature than most others. You can always skip certain parts and move on to the next chapter.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe the basic notions of probability, descriptive statistics and hypothesis testing
# - summarize data using R
# - perform basic calculations by hand and in R related to distributions, confidence intervals and hypothesis testing
# - understand the sources of variability in business data

# %% [markdown]
# #### 3.1 Summarizing data
#
# Data can be summarized in numerical and graphical ways. For univariate, i.e., 1-dimensional data, numerical summaries mostly concentrate on centrality and variability. The most common measure for centrality is the mean (`mean()` in R, also called average), equal to the sum of the values divided by the number. Other measures for centrality are the trimmed mean (by adding a second argument to the `mean()` function) and the median (`median()`). The trimmed mean ignores the lowest and highest values, the median is the "middle" value, for which 50% is lower and 50% is higher.
#
# For variability we mostly use the standard deviation (SD, `sd()` in R) or the variance (var, the square of the SD). They are defined later, but for both hold: the higher the value, the higher the variability.
#
# For example, the R datasets package contains a number of datasets of which `eurodist` is one. It contains distances (in km) between a number of major European cities. Then the R commands
#
# ```r
# > mean(eurodist); mean(eurodist,trim=0.1); median(eurodist)
# > sd(eurodist); var(eurodist)
# ```
#
# result in: 1505.1, 1422.9, 1311.5, 898.8 and 807813. Note that the second command computes the trimmed mean, ignoring the 10% lowest and highest values.
#
# **Exercise 3.1** Reproduce this calculation (you might need to load the datasets library). Do the same thing for a dataset consisting of all the same numbers. You can construct such a dataset with the function `rep()`. Now change a few of the numbers and look at the consequences.
#
# > **Box 3.1. The use of the mean**
# >
# > Very often we are interested in the average. However, this average might be highly influenced by a number of extreme points.
# >
# > For example, you might be interested in the average price of houses in a certain area, and see how it evolves over time. In many neighborhoods, this average is highly influenced by a small number of very expensive luxury houses. If you are interested in the price of a common house then the trimmed mean or the median might be a better choice.
#
# Next we consider graphical summaries, especially histograms and boxplots, made using the R functions `hist()` and `boxplot()`. Output for the `eurodist` dataset can be found in Figure 3.1. A histogram has different values grouped in buckets (here of length 500) on the horizontal axis and their frequencies on the vertical axis. We see, for example, that values between 1000 and 1500 km occur 52 times. A box plot is common in statistics and is essentially a 1-dimensional diagram in which the box is limited by the first and third quartile of the data (i.e., the points with 25% and 75% of the data below them, in R `quantile(eurodist,0.25)` and `quantile(eurodist,0.75)`), with the second quartile (the median) in the middle. There are different definitions for the horizontal lines (the whiskers) but the idea is that they show dispersion. Data points outside of the whiskers, the outliers, are shown as small circles.

# %% [markdown]
# ![Figure 3.1 (left): histogram of the eurodist dataset](images/lecture12_fig3.1-hist.png)
#
# ![Figure 3.1 (right): boxplot of the eurodist dataset](images/lecture12_fig3.1-boxplot.png)
#
# Figure 3.1: Histogram and boxplot

# %% [markdown]
# Regardless of your ultimate goal, it is always good to start with summarizing your data. This helps you to get a general impression of the data, its outliers, skewness, etc.
#
# **Exercise 3.2** Use the `AirPassengers` dataset from the datasets package. Compute all 5 quartiles, the average and the SD. Plot the histogram and the boxplot. You can do this in R and/or in Excel.
#
# > **Box 3.2. Skewness and outliers**
# >
# > From the example we see that the dataset is not symmetric, but skewed to the right with two outliers beyond 4000. This skewness to the right results in a mean that is bigger than the median. The outliers make the SD high.
# >
# > Data is rarely symmetric. Only in certain cases theory predicts symmetry, in other cases we hardly ever find it.
#
# > **Box 3.3. Summarizing data in Excel**
# >
# > Functions in Excel have partly different names than in R: average, median, percentile, stdev, and var. To construct a histogram you can use the "data analysis" add-in, which is unfortunately not available in all versions of Excel. You can also make the histogram by hand, by calculating first the number of data points per bucket, as illustrated in the example below.

# %% [markdown]
# ![Box 3.3: making a histogram by hand in Excel, calculating the number of data points per bucket](images/lecture12_box3.3-example.png)

# %% [markdown]
# #### 3.2 Probability theory and the binomial distribution
#
# In the previous section we analyzed data, which might be useful by itself. However, we might also consider the data to be outcomes of some experiment with uncertain, random outcomes. What can we say of the experiment on the basis of its outcomes? How can we define the experiment in a useful way? Probability theory gives the framework to answer this type of question. In probability, a random experiment is called a random variable (RV), often denoted with the letter $X$. An RV $X$ is different from a regular variable $x$ in the sense that it can take multiple values, according to its distribution. For example, if $X$ models the rolling of a die then it can take values $\{1, 2, \dots, 6\}$, each with probability $1/6$.
#
# Distributions come in two flavors: discrete and continuous. Discrete RVs can take a finite or countable number of values. For example, rolling a die (with possible outcomes $1, \dots, 6$), flipping a coin (0 or 1), or the number of arrivals to a service center (possibly $0, 1, 2, \dots$).
#
# Continuous RVs can take any value within a specified range, for example all values positive and negative (denoted by $\mathbb{R}$), only positive values ($\mathbb{R}^+$), or a range such as $[1, 5]$, the interval from 1 to 5. Examples are body heights and GDP of a country.
#
# In what follows, we discuss a number of often used distributions, starting with discrete ones. We start with the Bernoulli and binomial distributions. While discussing them we introduce some important concepts from probability theory.
#
# **The Bernoulli distribution**
#
# The Bernoulli distribution (named after the Swiss mathematician) can only take two values: 0 and 1. It models situations such as coin tossing. Random variables always have numbers as outcomes, therefore in the cointossing case "heads" and "tails" are translated to 0 and 1. A Bernoulli distributed RV $X$ is completely defined by the probability $p$ by which 1 occurs, written as $P(X=1) = p$, which should be read as "the probability that the RV $X$ is equal to 1 is equal to $p$".
#
# A common way to plot a distribution is by its cumulative distribution function (cdf), usually denoted as $F(x)$ or $F_X(x)$, and defined as $F_X(x) = P(X \le x)$. It follows that $F$ is a non-decreasing function with $F(-\infty) = 0$ and $F(\infty) = 1$. When $X$ is continuous $F$ is also continuous (hence the name). When $X$ is discrete, $F$ is a step function: it is constant between the values that can occur and at these points it makes jumps. $F_X$ for $X$ Bernoulli is plotted in Figure 3.2.

# %% [markdown]
# ![Figure 3.2: The cdf of the Bernoulli distribution](images/lecture12_fig3.2.png)

# %% [markdown]
# **Exercise 3.3** For the cdf below, give the possible outcomes and their probabilities.

# %% [markdown]
# ![Exercise 3.3: an uncaptioned cdf](images/lecture12_ex3.3-cdf.png)

# %% [markdown]
# In a way somewhat similar to the mean of a dataset, we can compute the expectation of a random variable: it is an average over all possible outcomes, weighted with the probabilities.
#
# **Example 3.1** Rolling a die has as possible outcomes $1, \dots, 6$, each with probability $1/6$. Its expectation is therefore $\frac{1}{6} \times (1 + \dots + 6) = 3.5$.
#
# In a similar way we can also compute the SD and variance of a RV. They are denoted as $EX$, $\sigma(X)$, and $\sigma^2(X)$, respectively. For the Bernoulli distribution their values are:
#
# $$
# EX = p, \qquad \sigma(X) = \sqrt{p(1-p)}, \qquad \sigma^2(X) = p(1-p). \tag{3.1}
# $$
#
# > **Box 3.4. Mathematical expectation and variance**
# >
# > The definition of the expectation for a discrete RV $X$ is as follows:
# >
# > $$
# > EX = \sum_{x:\, P(X=x)>0} x P(X=x).
# > $$
# >
# > For $X$ Bernoulli this gives $EX = 0 P(X=0) + 1 P(X=1) = p$. The variance is the expected quadratic difference from the expectation:
# >
# > $$
# > \sigma^2(X) = E(X - EX)^2 = \sum_{x:\, P(X=x)>0} (x - EX)^2 P(X=x).
# > $$
# >
# > For $X$ Bernoulli it is $\sigma^2(X) = (0-p)^2(1-p) + (1-p)^2 p = p - p^2$.
#
# The Bernoulli distribution is a special case of the binomial distribution, which we will discuss next. R functions for both distributions will be introduced after that.

# %% [markdown]
# **The binomial distribution**
#
# Suppose we repeat a 0/1 experiment $n$ times. We assume that they are independent, meaning that the outcome of one does not influence another. Let $N$ be the total number of 1s. Then $N$ has a so-called binomial distribution. The binomial distribution has two parameters: the success probability $p$, and $n$. For $n=10$ and $p=0.2$ its cdf is plotted in Figure 3.3. This plot was made using the following R command: `curve(pbinom(x,10,0.2))`.
#
# > **Box 3.5. Independence**
# >
# > Independence is a very important property. Often we repeat an experiment multiple times. For example, we try a new medication on multiple patients, or we observe multiple visitors to a webshop. Statistical independence states that the outcome of one experiment does not influence the other. We assume it quite often, because it makes the analysis much simpler, although it might not be completely true. For example, a webshop customer who purchased a product might leave a positive review, thereby increasing the purchase probability of future customers. In this situation, the experiments are not independent, although the purchase probability might have only changed very little.
# >
# > However, there are situations where we prefer not to have independence. For example, if we know the cancer type of each patient and the effect of a certain medication, then we might hope for dependence, i.e., a correlation between the type of cancer and the effect, in order to be able to give the right medication to the right patient. Similarly, you hope that attributes such as previous visits, age, time on website, etc., influence the conversion probability in order to be able to steer behavior of webshop visitors, for example by targeted advertising. These are typical examples of multivariate problems which we will discuss in later chapters.
#
# The name binomial comes from Newton's binomium, written as $\binom{n}{k}$. It gives the number of ways to select $k$ items out of $n$, is related to the Triangle of Pascal, and is part of the formula for $P(N=k)$. We will not go into the mathematical details, instead we discuss the R functions by which we can compute expressions such as $P(N=k)$, the probability of $k$ successes. There are 4 R functions:
#
# - `dbinom(k, n, p)`, which gives $P(N=k)$ for parameters $n$ and $p$;
# - `pbinom(k, n, p)`, which gives $P(N \le k)$ for parameters $n$ and $p$;
# - `rbinom(k, n, p)`, which gives $k$ random independent outcomes of $N$ with parameters $n$ and $p$;
# - `qbinom(q, n, p)`, which gives the inverse of the cdf: the number $k$ such that $P(N \le k)$ is equal or just above $q$.
#
# As an example, consider a school class with 30 kids who are randomly selected.

# %% [markdown]
# ![Figure 3.3: The cdf of the binomial distribution with n = 10 and p = 0.2](images/lecture12_fig3.3.png)

# %% [markdown]
# We assume that the probability of every child being male or female is exactly 50%. Then the probability of having 15 kids of each sex is `dbinom(15,30,0.5)` is equal to 14.4%. Having 10 or less girls has probability `pbinom(10,30,0.5)`, 5%.
#
# Generating 10 arbitrary classes by `rbinom(10,30,0.5)` leads to: 18 13 13 12 15 16 14 14 19 17.
#
# Explaining the use of `qbinom()` is a bit harder. Suppose we want to know the maximum number of girls to expect in 90% of the classes. Then `qbinom(0.9,30,0.5)` gives the answer: 19. Indeed, `pbinom(19, 30, 0.5)` $= P(N \le 19) \ge 0.9$ and `pbinom(18, 30, 0.5)` $= P(N \le 18) < 0.9$.
#
# The values given by `qbinom()` are also called percentiles. The 25th, 50th and 75th percentile are called quartiles; the 50th percentile is the median of the distribution.
#
# **Exercise 3.4** You roll a die 10 times. What is the probability that there are no 6s? Make a plot of the probability of $k$ 6s for $k \in \{0, \dots, 10\}$.
#
# Note that the R functions for the binomial distribution can be used as well for the Bernoulli distribution, by taking $n=1$.
#
# The formulas for the expectation, SD and variance are as follows:
#
# $$
# EN = np, \qquad \sigma(N) = \sqrt{np(1-p)}, \qquad \sigma^2(N) = np(1-p).
# $$
#
# Note the resemblance with the Bernoulli distribution. The reason for this is explained hereafter.
#
# **Example 3.2** You roll a die 10 times. Then you expect $10 \times \frac16 = 1.67$ times a 6, and the standard deviation of the number of 6s is $\sqrt{10 \times \frac16 \times \frac56} = 1.18$.

# %% [markdown]
# **Sums of independent random variables**
#
# Often we are interested in sums or averages of usually independent random variables. We will discuss two aspects of this: the distribution of sums and the expectation and SD of sums and averages. Let us start with the latter. For random variables, the following rules hold:
#
# - the expectation of the sum is the sum of the expectations;
# - the expectation of a constant times a RV is the constant times the expectation.
#
# In mathematical terms this is equivalent to:
#
# $$
# E(X+Y) = EX + EY \qquad \text{and} \qquad E(cX) = cEX.
# $$
#
# The first formula explains why $EN = nEX$ for $N$ binomial and $X$ Bernoulli. It is interesting to note that $X$ and $Y$ do not even have to be independent!
#
# **Example 3.3** $X$ and $Y$ denote next year's profit of two business units of a company. The total expected profit is $EX + EY$. $X$ and $Y$ are allowed to be dependent, for example on the same yet unknown interest rate.
#
# Business unit 1 has to pay 30% taxes. Its expected tax payment is $0.3EX$.
#
# For the SD and the variance things are a bit more complicated:
#
# - the variance of the sum is the sum of the variances;
# - the SD of a constant times the RV is the constant times the SD.
#
# In mathematical terms:
#
# $$
# \sigma^2(X+Y) = \sigma^2(X) + \sigma^2(Y) \qquad \text{and} \qquad \sigma(cX) = c\sigma(X).
# $$
#
# Here $X$ and $Y$ need to be independent (in fact, uncorrelated suffices). Because $\sigma(X) = \sqrt{\sigma^2(X)}$ and $\sigma^2(X) = (\sigma(X))^2$ we also have:
#
# $$
# \sigma(X+Y) = \sqrt{\sigma^2(X) + \sigma^2(Y)} \qquad \text{and} \qquad \sigma^2(cX) = c^2\sigma^2(X).
# $$
#
# **Example 3.4** We continue with Example 3.3. Suppose that the numbers are as follows: $EX=8$, $\sigma(X)=3$, $EY=12$, $\sigma(Y)=4$ (all in M€). Let us assume that $X$ and $Y$ are independent. Then $E(X+Y) = 20$ and
#
# $$
# \sigma(X+Y) = \sqrt{3^2+4^2} = 5,
# $$
#
# which is considerably smaller than the sums of the SDs, which is 7. This is exactly why investors "spread" their risk: the risk of a portfolio is smaller than the sum of the risks. The loss on one investment might be compensated by the others.
#
# The caveat is in the independence: in times of economic downturn the values of assets tend all to go down contradicting the independence.
#
# Thus, in these expressions it is crucial not to replace SD by variance and vice versa! The mathematical proofs of all expressions are not very difficult but require some experience with manipulating summations and integrals.
#
# **Averages**
#
# Let us now consider averages. To do so, let $X_1, \dots, X_n$ be $n$ independent RVs with the same distribution. Thus they have the same expectation and SD. Define $\bar X$ as the average of $X_1, \dots, X_n$:
#
# $$
# \bar X = \frac{X_1 + \dots + X_n}{n}. \tag{3.2}
# $$
#
# Averages play an extremely important role in statistics, because for high $n$ they tend to the expectation, as we will see below. Therefore averages are used as estimators in cases where the expectation is unknown. We give the expectation and SD of the average:
#
# $$
# E\bar X = EX_1 \qquad \text{and} \qquad \sigma(\bar X) = \sigma(X_1)/\sqrt n. \tag{3.3}
# $$
#
# **Exercise 3.5** Verify the correctness of these expressions using the rules presented earlier.
#
# How should we interpret these results? Our (statistical) experiment consists of $n$ observations. The outcome is likely (in the statistical sense: with high probability) to be close to the expectation. As we perform more observations, then the outcome is likely to be closer to the expectation: the SD decreases, because we divide by $\sqrt n$. Note that the function $\sqrt n$ increases slowly, therefore if we want to increase the accuracy of the outcome, we have to include many more observations! When we perform infinitely many observations (a mathematical abstraction, we cannot perform that many observations in practice) then the SD becomes 0 and we find exactly the expectation.
#
# This result is known as the law of large numbers (LLN). It is the basis of statistics: if we take a sufficiently large sample from a population then the average is a reliable estimator of the expectation. In the context of hypothesis testing, we will discuss when a sample is "sufficiently large". Note that it is assumed that all observations have the same distribution and are independent. In the design of an experiment, this means that there is no selection bias, every observation should be representative for the whole population.
#
# **Exercise 3.6** Sample 1000 times from a Bernoulli distribution with success probability 0.5. For every $n$, take the average over the first $n$ numbers and make a plot of this as a function of $n$. How does this illustrate the LLN?
#
# Sometimes we are interested in the distribution of a sum of distributions. This is for example the case if we roll a die twice and we want to know the probability that the sum of the outcomes is 10. In such a case there are multiple approaches:
#
# - in certain cases we have theoretical results concerning sums. The binomial distribution is an excellent illustration of this: it is itself a sum of Bernoulli distributions having the same $p$, and for the same reason sums of binomial distributions are again binomial, a long as they are independent and have the same $p$. Another example is the normal distribution, which will be discussed later on: sums of normals are again normal;
# - for the majority of sums no mathematical expression is known. In that case we can often do a numerical calculation to compute the joint distribution;
# - a simple and intuitive alternative is sampling or simulation, based on the LLN. You simply sample every component of the sum many times and you take the sums. For example, `rbinom(100,20,0.2)+rbinom(100,10,0.2)` gives 100 samples of a binomial distribution with parameters 30 and 0.2. If you change one of the $p$s the sum is not binomial anymore. However, the method can still be used.
#
# **Exercise 3.7** a. Simulate the sum of two dice many times and use this to approximate the probability that the sum of the outcomes is 10.
#
# b. Determine the probability by calculating all possible outcomes that lead to 10 and their probabilities.

# %% [markdown]
# #### 3.3 Other distributions and the central limit theorem
#
# In the previous section we introduced the binomial distribution and used it to introduce some important concepts from probability theory. In this section we introduce some other well-known distributions.
#
# **The Poisson distribution**
#
# The Poisson distribution is used in practice to model customer arrivals to service centers, such as visits to web sites or calls to a call center. Arrivals to these centers show fluctuations from minute to minute. Statistical analysis shows that they often follow a Poisson distribution.
#
# The Poisson distribution (named after the French mathematician who invented it) has a single parameter, often indicated with the Greek letter $\lambda$. A Poisson distributed RV $N$ has the special property that $EN = \sigma^2(N) = \lambda$. Sums of Poisson distributions are again Poisson distributions with the sum of the parameters.
#
# **Exercise 3.8** For a fixed $\lambda$, consider binomial distributions $N_n$ with $n$ experiments and success probability $\lambda/n$. Look up the formula for the binomial distribution (e.g., at Wikipedia) and show that $\lim_{n \to \infty} P(N_n=k)$ equals the Poisson distribution. (This exercise requires knowledge of calculus, the mathematical field that includes integration and limits.)
#
# The R functions for the Poisson distribution are `dpois`, `ppois`, `qpois`, and `rpois`. Their definition is similar to those of the binomial distribution (with 1 parameter less). Indeed, every distribution defined in R has functions of the form `dxxx`, `pxxx`, `qxxx`, and `rxxx`, with `xxx` the abbreviation of the name of the distribution.
#
# **Exercise 3.9** a. Compute the SD of 1000 samples of a Poisson distribution with $\lambda=10$. Is the answer as expected?
#
# b. Plot the cdf's of the Poisson distributions with $\lambda = 1, 5$ and 20 in a single figure.
#
# **Exercise 3.10** A web server can handle 100 requests per minute. Additional demand is lost. Arrivals occur according to Poisson distribution with average 90. Estimate the percentage of requests lost.
#
# Hint: you can solve this problem by sampling the demand a number of times and calculating for every sample the number of requests lost. You can also obtain the exact result by using the distribution but this is more difficult and less intuitive.
#
# **The uniform distribution**
#
# The uniform distribution is the first continuous distribution we discuss. It has a minimum and a maximum, commonly denoted with $a$ and $b$. The defining feature of the uniform distribution is that every interval between $a$ and $b$ of the same length is equally likely. For this reason the cfd, given in Figure 3.4, increases linearly in the interval $[a,b]$ from 0 to 1.

# %% [markdown]
# ![Figure 3.4: The cdf of the uniform distribution with a = 1 and b = 5](images/lecture12_fig3.4.png)

# %% [markdown]
# For $U$ uniformly distributed on $[a,b]$, the expectation and standard deviation are as follows:
#
# $$
# EU = \frac{a+b}{2} \qquad \text{and} \qquad \sigma(U) = \frac{b-a}{\sqrt{12}}.
# $$
#
# The R functions are `dunif`, `punif`, `qunif` and `runif`. E.g., `punif(1,0,3)` gives $1/3$ and `qunif(2/3,0,3)` is equal to 2. See Box 3.6 for the interpretation of `dunif`.
#
# **Probabilities of eventualities**
#
# Note that `punif(u,a,b)` gives $P(U \le u)$ with $U$ uniform with parameters $a$ and $b$. However, sometimes we are interested in probabilities of other intervals, such as $P(U>u)$ or $P(U \in [u,v])$. These expressions can be derived from `punif`. From $P(U \le u) + P(U > u) = 1$ it follows that
#
# $$
# P(U>u) = 1 - P(U \le u) = 1 - \texttt{punif(u,a,b)}.
# $$
#
# Similarly,
#
# $$
# P(U \in [u,v]) = P(U \le v) - P(U \le u) = \texttt{punif(v,a,b)} - \texttt{punif(u,a,b)}.
# $$
#
# Note that for continuous distributions $P(X \le x) = P(X < x)$. The explanation can be found in Box 3.6. For discrete distributions this does matter! Intervals like $A = [u,v]$ are called eventualities, $P(A)$ is its probability. As part of the fundamentals of probability more complicated sets $A$ are studied. We will stay far away from this type of mathematical sophistication.
#
# **Exercise 3.11** For $U$ uniformly distributed with parameters 0 and 2, determine by hand $P(U \in [0.5,1])$. Check your answer by sampling in R many times from $U$ using `runif` and by using `punif`.
#
# **The normal distribution**
#
# We continue our focus on distributions with the most famous of them all: the normal or Gaussian (after the German scientist) distribution. The normal distribution has two parameters: $\mu$ and $\sigma$, the expectation and the SD. Be careful: some tools require you to enter the SD, some require the variance. Of course, they are not the same, unless $\sigma=1$ (or 0, but then there is no variability: the degenerate distribution that has as outcome $\mu$ with probability 1).
#
# > **Box 3.6. Probability of a single outcome**
# >
# > A surprising and counter-intuitive feature of continuous distributions is that every possible outcome has probability 0. This is because there are infinitely many points in an interval such as $[a,b]$. If they all had a positive probability of occurring then they would sum up to more than 1. Indeed, if we measure all people in the world up to 10 decimals then nobody would be exactly 1m80. However, we can attribute a probability to intervals, such as all people having a length between 1m80 and 1m81.
# >
# > For discrete distributions, the R function `pxxx` gave the probability of a point. For continuous distribution, the definition is different: it gives the so-called density. Integrating the density over the real numbers gives 1, just as all probabilities of a discrete distribution sum up to 1.

# %% [markdown]
# The normal distribution is well-known for its symmetric bell-shaped density, which is plotted for two distributions in the left plot of Figure 3.5. The corresponding cdf's (which are easier to interpret) are in the plot on the right. In the figure we used the common notation $N(\mu,\sigma^2)$ for normal distributions. Note the usage of $\sigma^2$: thus $N(3,4)$ has SD 2. The $N(0,1)$ is the standard normal distribution. The R function for the normal distribution are `dnorm`, `pnorm`, etc.
#
# **Exercise 3.12** a. Compute $P(X \le 0)$ for $X$ standard normal.
#
# b. Compute $P(X \ge 1)$ and $P(0 \le X \le 1)$.
#
# c. Give the 95th percentile of the standard normal distribution.
#
# Every normal distribution can be derived from the standard normal. If $X \sim N(0,1)$ (meaning that $X$ is N(0,1) distributed), then $Y = \mu + \sigma X \sim N(\mu,\sigma^2)$.
#
# **Exercise 3.13** a. Compute $P(X \ge 0)$ for an $N(-2,10)$ distribution. Do this directly and using the $N(0,1)$ distribution.

# %% [markdown]
# ![Figure 3.5: Densities and cdf of the normal distributions N(0,1) (solid) and N(3,4) (dashed)](images/lecture12_fig3.5.png)

# %% [markdown]
# b. Suppose that $x_1, \dots, x_n$ are samples of a $N(\mu,\sigma^2)$ distribution. How can you turn them into samples of a standard normal distribution?
#
# Let $X$ and $Y$ be independent normally distributed RVs. Because of the formulas on page 39 we have:
#
# $$
# E(X+Y) = EX + EY \qquad \text{and} \qquad \sigma^2(X+Y) = \sigma^2(X) + \sigma^2(Y).
# $$
#
# Moreover, the normal distribution has a very special property: sums of normal distributions have normal distributions. Note that not all distributions have this property. The normal and binomial distributions (with the same success probabilities) have this property, other distributions such as the uniform have not.
#
# **Exercise 3.14** Sample 10000 times from 2 normally distributed RVs, make a histogram of the sums and convince yourself that the statement above is true. How about $X - Y$? Can you explain this? Do the same thing for two uniform distributions.
#
# **Central limit theorem**
#
# Earlier, we saw that sums of normal distributions have a normal distribution. But there is more to it: all sums of independent RVs tend to look like normal distributions! For example, if you sum 10 uniform RVs, then the result looks pretty much like a normal distribution. The same holds for averages, as it is just a sum divided by a constant. Recall Equation (3.3) on page 40: $E\bar X = EX_1$ and $\sigma(\bar X) = \sigma(X_1)/\sqrt n$. Thus, as $n$ increases, $\bar X$ looks more and more like a normal distribution which is more and more concentrated around the mean $EX_1$. This is called the central limit theorem (CLT). It is illustrated in Figure 3.6. We see that already the distribution of the average of 10 uniform distributions has the bell shape of the density of a normal distribution.

# %% [markdown]
# ![Figure 3.6: Illustration of the CLT: histograms of averages of 1, 2, 5 and 10 uniform(0,1) realizations](images/lecture12_fig3.6.png)

# %% [markdown]
# > **Box 3.7. Formal statement of the CLT**
# >
# > The formal statement of the CLT is as follows. First we rescale $\bar X$:
# >
# > $$
# > Z_n = \frac{\sqrt n (\bar X - EX_1)}{\sigma}.
# > $$
# >
# > Now $Z_n$ has $EZ_n=0$ and $\sigma^2(Z_n)=1$. Thus subtracting $EX_1$ moved the average to 0 and blowing it up with $\sqrt n$ avoided it to disappear in 0. The CLT states that $Z_n$ converges to a standard normal distribution, i.e., as $n$ increases it looks more and more like a normal distribution which it reaches at $\infty$.
# >
# > To make it really formal you have to define convergence of distributions. We won't go into that level of detail.
#
# Averages play an important role in statistics, because we use them as estimators for the expectation (thanks to the LLN). To say something about the accuracy of this estimator, we can use normal distributions (thanks to the CLT). Thus computations with normal distributions are important in statistics. The following rules of thumb are often used, for $X \sim N(\mu,\sigma^2)$:
#
# - $P(\mu - \sigma \le X \le \mu + \sigma) \approx 68\%$;
# - $P(\mu - 2\sigma \le X \le \mu + 2\sigma) \approx 95\%$.
#
# The rule is illustrated in Table 3.7.

# %% [markdown]
# ![Figure 3.7: Rule of thumb for the normal distribution (source: Wikipedia)](images/lecture12_fig3.7.png)

# %% [markdown]
# **Exercise 3.15** Reproduce these numbers using `qnorm` in R and `NORM.INV` in Excel.
#
# A common error is to apply this rule to all kinds of data and distributions. The answers that you will get are wrong! Sometimes analysts first calculate the average and SD, to use the rule of thumb to find for example the 95th percentile. Not only do they get the wrong answer, there is also a simpler procedure: In a dataset of say 1000 points, they could have taken right away the 950th largest number.
#
# **Exercise 3.16** A hospital performs knee surgery routinely in one of its operating rooms. An operation takes on average 50 minutes with a SD of 20, including cleaning, changing, etc. A session consists of 8 operations, a block of 7 hours is reserved for it. Approximate the probability that the 8 operations take more than the reserved session time. Do this in two ways:
#
# - use a normal approximation;
# - simulate the session many times, assuming that the operations have a uniform distribution.
#
# Note that for the latter exercise you first need to determine the parameters of the uniform distribution.
#
# In the previous exercise, we saw the mathematical theory at work: for theoretical reasons we used a normal distribution, which gave us the same result as the one based on simulation.
#
# **The lognormal distribution**
#
# If data is positive and continuous then they often follow a lognormal distribution. Examples are durations of surgery or length of telephone calls. Lognormal RVs are of the form $e^X$ with $X$ a normally distributed RV and $e$ a mathematical constant, $e \approx 2.7$. A typical density and cdf can be found in Figure 3.8. The distribution is clearly skewed to the right. The plots can be made with the following R commands: `curve(dlnorm(x,3,0.5))` and `curve(plnorm(x,3,0.5))`. Note that 3 and 0.5 are the mean and SD at the logscale, of the underlying normal distribution. The real mean and SD are quite complicated formulas of the parameters. Probabilities and quantiles however can easily be derived from the underlying normal distribution. For example, `plnorm(x) = pnorm(log(x))`. `LOGNORM.DIST` and `LOGNORM.INV` are the Excel equivalents of `plnorm` and `qlnorm`.

# %% [markdown]
# ![Figure 3.8: Density and cdf of a lognormal distribution](images/lecture12_fig3.8.png)

# %% [markdown]
# **Exercise 3.17** Find the median of the distribution of Figure 3.8 in 3 ways: using the figure, using `qlnorm`, and using `qnorm`.
#
# **Exercise 3.18** Make a histogram of the product of 2 lognormal distributions. Do you recognize the distribution that you find? What could be the reason? (This requires some mathematical insights.)
#
# **Exercise 3.19** Sample from a lognormal distribution with mean 10 and SD 5. Note that you first have to compute $\mu$ and $\sigma$ of the underlying normal distribution, the formulas can for example be found on Wikipedia. Check that the sample has indeed the right mean and SD.

# %% [markdown]
# #### 3.4 Parameter estimation
#
# The goal of statistics is to infer unknown information from data. For this reason, we sometimes talk of inferential statistics, to differentiate from (statistical) data analysis. Data contains noise, for this reason we have to differentiate between noise and signal.
#
# There are basically two ways to proceed. Sometimes you have a hypothesis concerning the experiment that you want to test, for example if a coin is biased. Then we can use hypothesis testing. Sometimes we want a reliable estimator of some parameter, such as the average length of a population. This estimator can take the form of an interval, called a confidence interval.
#
# **Confidence intervals**
#
# In both hypothesis testing and confidence intervals, the concept of the sample mean has a central place, defined in Equation (3.2). Suppose, as an example, we have measured the height of 100 arbitrary adult Dutch men and women and the average is 178 cm. Evidently, the average height of the whole population is not exactly 178 cm, because we took a sample, and we therefore certainly introduced an error. How big is this error? Can we construct an interval in which the true value falls with a certain level of confidence? This confidence interval (CI) is constructed as follows. Suppose that the height of the Dutch follows a distribution $X$ with mean $\mu$ and SD $\sigma$. Then, according to (3.3) and the CLT, $\bar X$ is approximately normal distributed with $\sigma(\bar X) = \sigma/\sqrt n$. Then a 95% CI is given by $[\bar X - 2\sigma/\sqrt n, \bar X + 2\sigma/\sqrt n]$, using the rule of thumb of Figure 3.7. However, we cannot compute this interval: in general we do not know $\sigma$. Therefore, we need to estimate it, by the sample SD, which is given by
#
# $$
# S = \sqrt{\frac{\sum_{i=1}^n (X_i - \bar X)^2}{n-1}}.
# $$
#
# **Exercise 3.20** One would expect $n$ instead of $n-1$ in the denominator of $S$. The reason is that $S$ in its current form is an unbiased estimator, i.e., $ES = \sigma$. Show this. (This exercise requires quite some mathematical skills.)
#
# In our example, suppose that the sample SD of the height of the Dutch is 5 cm. Then the CI for the average height becomes $[178 - 2 \times 5/\sqrt{100}, 178 + 2 \times 5/\sqrt{100}] = [177, 179]$.
#
# > **Box 3.8. Interpretation of CI**
# >
# > A CI is commonly interpreted as an interval in which the true value falls with a certain probability. Correctly speaking, this is wrong: $\mu$ has an unknown but fixed value so it is within an interval or not. The correct interpretation of a CI is as follows: if you repeat an experiment multiple times and you create a CI every time, then in $\alpha$ (the confidence level) cases $\mu$ is inside the interval.
#
# **The CI made precise**
#
# We said that the 95% CI for the mean is given by $[\bar X - 2S/\sqrt n, \bar X + 2S/\sqrt n]$. However, this is not completely true: the 97.5% quantile of the standard normal distribution is 1.96, which can be verified in R with `qnorm(0.975)`.
#
# But there is more to it than that. Because we do not know $\sigma$ we replaced $(\bar X-\mu)/(\sqrt n \sigma)$ by $(\bar X-\mu)/(\sqrt n S)$. While the former has a standard normal distribution, the latter doesn't, because $S$ is a random variable. The true distribution of $(\bar X-\mu)/(\sqrt n S)$ is called Student's t-distribution with $n-1$ degrees of freedom. Thus we should replace 2 by `qt(0.975,99)`, 1.98. We see that the CI gets slightly larger, from 1.96 to 1.98. For larger $n$, the difference is even smaller, thus in almost all cases 2 is a very good approximation.
#
# **Exercise 3.21** A sample of 200 entries has average 9.8 and sample SD 5.4. Calculate a 90% CI for the population mean.
#
# **Hypothesis testing**
#
# In a hypothesis test, we reject a hypothesis when the outcomes are very unlikely when the null hypothesis would be true. As an example, assume we want to test whether a coin is unbiased. We throw it 100 times and it comes up heads 62 times. What can we conclude? The standard procedure is to compute the probability of the outcome or more extreme under the null hypothesis, which is called the p-value. When this p-value is below the significance level (often 5%) then we reject the null hypothesis in favor of the alternative hypothesis. The probability of 62 or more is 1% (`1-pbinom(61,100,0.5)`). This is less than 5%, therefore the null hypothesis is rejected: we have sufficient statistical evidence to conclude that the coin is biased.
#
# We actually tested whether heads is more likely to come up then tails. This is called a one-sided test. For a two-sided test (biased or unbiased, no matter if heads or tails is overrepresented) we have to reserve 2.5% for both sides. Because 1% < 2.5% we still reject. Alternatively, we could multiply the probability by 2. This is actually the standard way to calculate the p-value.
#
# R has a build-in test for this situation: `binom.test`, which should be used as follows: `binom.test(62,100)`. We get as p-value 2% because the default test is 2-sided.
#
# **Exercise 3.22** You roll a die 12 times.
#
# a. How many sixed do you expect?
#
# b. Assume that no sixes occurred. Can you conclude that the die is biased?
#
# In the exercise, does the p-value mean that we can conclude that the die is not biased? The answer is no: if the p-value is higher than 5% no conclusion can be drawn. In this sense, hypothesis testing is asymmetric: only when the p-value is smaller than the significance level can a conclusion be drawn.
#
# When we take a 0/1 sample then we know that the sum is binomial. In general, we do not know the distribution of a sum of RVs and we have to use the CLT. The resulting test for the mean is the t-test, again using Student's t-distribution, although a normal approximation is simpler and gives nearly the same results.
#
# As an example, take the average height of adults. Although data is partly unreliable, the average overall world-wide height of adults is around 174 cm. Can we conclude that the Dutch are taller than the worldwide average based on our sample with size 100? Our test is as follows:
#
# $$
# H_0: \mu = \mu_0 = 174 \quad \text{versus} \quad H_1: \mu > \mu_0 = 174.
# $$
#
# From `1-pnorm(sqrt(100)*(178-174)/5)` (or, using the mathematically correct t-distribution, `1-pt(sqrt(100)*(178-174)/5,99)`) it follows that the p-value is very small. The conclusion is therefore that the null-hypothesis is rejected and that the Dutch are taller than the world average. R also contains commands for directly executing tests. If the data is entered as an array called `"data"` in R, then you can use `t.test(data,mu=174,alt="g")`, where `"g"` refers to the alternative hypothesis which is not two-sided, but "greater".
#
# **Exercise 3.23** In this exercise we use the `beaver1` dataset in the R datasets package, which you should install first. Use a t-test to check whether this beaver's average body temperature is significantly different from the average human body temperature (37.3 Celcius). Also determine the p-value directly by computing mean and SD and draw your conclusions.
#
# **Other univariate tests**
#
# There are other situations where tests on univariate data (the subject of this chapter) can be very helpful. On the internet, there is ample information on many different tests. Therefore we will not go into all details, we will just give a few of the most often used tests.
#
# **Comparing two samples**
#
# A frequently occurring task is comparing two datasets to see if their means are equal or not. There are two possibilities: the datasets consists of matched pairs or not. In the case of matched pairs, there is a dependence between the pairs in the two datasets. This is for example the case if you compare incomes between men and women in the same household: you often find high- and low-income couples. A case where the datasets are not matched and perhaps even of different sizes, is when you compare incomes of inhabitants of different cities. To analyse the difference in means we have to determine the difference of the averages. Consider random samples $X_i$ and $Y_j$, with sizes $n$, and $m$. As test statistic $T$ we take
#
# $$
# T = \frac{\bar X - \bar Y}{\sqrt{\frac{S_X^2}{n} + \frac{S_Y^2}{m}}},
# $$
#
# where $S_X$ and $S_Y$ are the sample SDs. Note that the denominator is equal to $\sigma(\bar X - \bar Y)$ and that $T$ is approximately standard normal, under the null hypothesis. Now we can use the normal distribution to perform our test, or rely on the build-in function of R.
#
# **Exercise 3.24** Now we compare the average body temperatures of `beaver1` and `beaver2` in the datasets library. Compute the test statistic and determine whether we reject the null hypothesis that the temperatures are equal. Do this also using the R command `t.test`.
#
# **Testing for a distribution**
#
# Often we are interested to know whether data comes from a certain distribution. However, a test can never confirm that data comes from a certain distribution; it can only tell us how unlikely it is. Different tests consider different aspects of distributions, so it might occur, for exactly the same null hypothesis, but a different statistic, that one test rejects the null hypothesis while another test does not reject.
#
# The Shapiro-Wilk test is a test for normality, with `shapiro.test` the R command. You do not need to specify the parameters of the normal distribution.
#
# > **Box 3.9. Q-Q plots**
# >
# > It is a good habit to take a careful look at the data before testing for a distribution. One way to see graphically if a distribution might fit the data is by making a Q-Q plot. In a Q-Q plot, we plot the quantiles of the data against those of a certain distribution. A close to straight line indicates that your data might well come from that distribution.
# >
# > As an example, see the left plot of the figure below, generated by `qqnorm(beaver1$temp)`. In the middle, the line is quite straight; the deviations at the sides indicate outliers. This is confirmed by the histogram on the right. A further confirmation comes from the Shapiro-Wilk test, executed by `shapiro.test(beaver1$temp)`: normality is rejected.

# %% [markdown]
# ![Box 3.9 (left): Q-Q plot of the beaver1 temperature data](images/lecture12_box3.9-a.png)
#
# ![Box 3.9 (right): histogram of the beaver1 temperature data](images/lecture12_box3.9-b.png)

# %% [markdown]
# > Note that using `qqplot` any two datasets or distributions can be compared.
#
# Another test is the Kolmogorov-Smirnov test, `ks.test` in R. It can be used in two ways: to find out if 2 datasets come from the same (continuous) distribution, and to test whether a dataset comes from a given distribution, which has to be specified including its parameters.
#
# **Exercise 3.25** We use the `Nile` dataset in the datasets library.
#
# a. Plot the histogram, boxplot and normal Q-Q plot. Does it look normal?
#
# b. Test for normality.
#
# c. Make a CI for the mean.
#
# d. Split the dataset in two by looking at the first 50 and the last 50 numbers. Is there a significant difference in average?

# %% [markdown]
# #### 3.5 Additional reading
#
# There are many books on probability theory. An accessible introduction is Ross [34]. The same holds for statistics. An accessible introduction is Triola [41].
#
# > **Box 3.10. Bayesian statistics**
# >
# > Central in Bayesian statistics is a distribution on the unknown parameter, such as the mean. This is in contrast with the frequentist approach which we discussed so far: there the unknown parameter was unknown but fixed. This distribution on the parameter is updated every time a new observation is made. From the a priori distribution we go, using Bayes' rule, to the a posteriori distribution. This can be done numerically, but for certain distributions analytical results are known.
# >
# > As an example, consider throwing a possibly biased coin. As initial distribution on the unknown success parameter, we choose the uniform distribution on $[0,1]$. Every time we throw the coin, we adapt this distribution using Bayes' rule. The resulting distributions are all so-called beta distributions. In the figure below we see a number of beta distributions. The top-left figure shows the uniform distribution. After two successes, the posterior distribution is as in the top-right, after 5 successes and 2 failures as in the left-bottom, and after 62 successes and 38 failures as in the right-bottom figure. The equivalent of a CI is a credible interval. A 95% credible interval of the last distribution is $[0.52, 0.71]$. This interval was computed with the R commands `qbeta(0.025,63,39)` and `qbeta(0.975,63,39)`.

# %% [markdown]
# ![Box 3.10: a sequence of beta distributions illustrating Bayesian updating (uniform prior; after 2 successes; after 5 successes and 2 failures; after 62 successes and 38 failures)](images/lecture12_box3.10.png)

# %% [markdown]
# ### Chapter 5 — Simulation
#
# Simulation can be used in cases where we know exactly how the attributes or components of some system interact to give a dependent value or output. Therefore simulation is a purely predictive method by which we can model any form of dependency. However, contrary to machine-learning models, the input is random. Simulation determines the random impact of the input on the output. For example, we can quantify the impact on waiting times of having an additional cashier in a supermarket or an additional lane in a road network. In these examples, the randomness comes from the unknown behavior of the customers or drivers.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe Monte Carlo and discrete-event simulation
# - perform Monte Carlo simulations in R and Excel
# - translate business simulation problems into simulation models
# - reflect on the usefulness of simulation in practice

# %% [markdown]
# #### 5.1 Monte Carlo simulation
#
# Suppose there are $k$ inputs and some known function $g$ that relates the inputs to the output, i.e., for inputs $x_1, \dots, x_k$ the output is $y = g(x_1, \dots, x_k)$. The central question in simulation is: can we give a reliable estimate of the expected output $EY = Eg(X_1, \dots, X_k)$? We assume that $X_1, \dots, X_k$ are know random variables which we often assume to be independent. We estimate $EY$ by repeatedly sampling from $(X_1, \dots, X_k)$ and computing $g$. Let us call the outcomes $Y_1, \dots, Y_n$. Then, according to the law of large numbers (see page 40), the average $\bar Y$ is an estimator for $EY$.
#
# **Example 5.1** A project consists of a number of activities with precedence constraints: an activity can only be started when all the preceding ones are finished. For example, the roof of a house can only be constructed when the walls are finished. Clearly, project planning is an important part of project management.
#
# The essence of projects is that each project is different. Therefore we cannot predict activity durations with certainty (as we can in manufacturing, for example following the lean approach). As a simple example, suppose we have two parallel activities followed by a third activity. Then the duration of the project is $g(x_1,x_2,x_3) = \max\{x_1,x_2\} + x_3$. Assume all three durations have uniform $[0,2]$ distributions (which have mean equal to 1). Then we can approximate the expected duration of the project with the following R command:
#
# ```r
# > mean(pmax(runif(n,0,2),runif(n,0,2))+runif(n,0,2))
# ```
#
# with $n$ equal to say 10000. With seed 0 this gives 2.33 as answer.
#
# Note that this is substantially more than 2, which would have been the answer in the case of deterministic durations with the same mean. Mistaking the duration of the means for the mean of the durations is a common mistake, called the strong form of the flaw of averages by Savage [36].
#
# It is logical to question the accuracy of the answer. In simulation it is common to construct a confidence interval for the outcome. As shown on page 49, a 95% CI is given by $[m - 2s/\sqrt n, m + 2s/\sqrt n]$, with $m$ the average outcome and $s$ its SD.
#
# **Example 5.2** Based on the same durations we can compute the sample SD of the small example project: 0.75. The CI is $[2.31, 2.34]$.
#
# Note that, as long as the simulations are relatively simple, we can take the sample size $n$ as big as we like and thereby obtain an arbitrarily accurate answer.
#
# **Exercise 5.1** a. Explain the difference between the R functions `max` and `pmax`.
#
# b. Using R, simulate the project of Figure 6.5 on page 92, with all activities having uniform distributions with mean as indicated and width $(b-a)$ equal to 2. Compute a CI.
#
# > **Box 5.1. Simulation in Excel**
# >
# > Microsoft Excel or comparable spreadsheets are less appropriate for simulation. By using F9 the sheet is recalculated, and all random variables are sampled again. However, this does not allow us to compute a CI. This can be done by putting the whole simulation on 1 row and copying that row many times. Add-ins to Excel exist (such as Crystal Ball) that add simulation functionality to Excel.
#
# c. Simulate the project again with all activities having lognormal distributions with mean as indicated and SD 1. Note that the mean and SD of the underlying normal distributions first need to be computed (see also Exercise 3.19).
#
# **Exercise 5.2** Repeat parts b) and c) of the previous exercise using Excel.
#
# > **Box 5.2. Tail probabilities**
# >
# > Sometimes we are not interested in the expectation of $g(X_1, \dots, X_k)$, but in probabilities of the form $P(g(X_1,\dots,X_k) \ge \alpha)$. However, a probability can be written as an expectation:
# >
# > $$
# > P(g(X) \ge \alpha) = \sum_{g(x) \ge \alpha} P(X=x) = \sum I\{g(x) \ge \alpha\} P(X=x) = EI(g(X) \ge \alpha),
# > $$
# >
# > with $I$ a special function, which is 1 when the argument is true and 0 otherwise, called the indicator function. Thus we end up estimating the expectation of the 0/1 function $I(g(X) \ge \alpha)$. For example, in the project planning example we are interested in the fraction of times that the project takes more than 2 time units. In R we compute this as follows:
# >
# > ```r
# > > durations=pmax(runif(n,0,2),runif(n,0,2))+runif(n,0,2)
# > > m=mean(durations>2);s=sd(durations>2)
# > > c(m-2*s/sqrt(n),m+2*s/sqrt(n))
# > ```
# >
# > This gives as CI $[0.65, 0.67]$.
#
# **Exercise 5.3** The budget of most companies are set without taking the variability of the numbers into account. Consider a simple budget:
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
# c. Determine the probability of a loss and a CI of this probability. Hint: first read Box 5.2 on "tail probabilities".
#
# > **Box 5.3. Machine learning versus simulation**
# >
# > Most machine-learning methods consist of two phases: first a descriptive phase in which a model is learned, within a certain class of models, such as linear with normally distributed noise. The second phase is the predictive part in which the output for new, deterministic, inputs is approximated. Simulation only consists of a predictive part, without restrictions on the model. It quantifies the consequences of uncertainty on the input to the output. Simulation can also be used for the predictive phase of a machine-learning model in case the input is random.

# %% [markdown]
# #### 5.2 Discrete-event simulation
#
# Sometimes a system is too complex to be modeled by some function $g$. This is especially the case when it concerns a process that evolves randomly over time. In such a situation discrete-event simulation (DES) is required. A central concept in DES is the state, which changes at random points in time. The state is discrete in nature, hence the name. The output or performance is usually a function of the state, averaged over time. The simulation is often terminated after a fixed amount of time, or when some condition is satisfied. We are usually interested in the expectation of the output measure. As the output of every run is a random variable, we can calculate a CI in the same way as we did for the Monte Carlo simulation.
#
# **Example 5.3** The evolution of waiting queues in a service center or the inventory positions in a warehouse are typical examples of processes that can be modeled using DES. Examples in other areas are the evolution of a disease in a body or the state of a communication network.
#
# In these examples, the state refers to the number of customers in the service center, the stock in the warehouse, the extent to which the disease has progressed, or the numbers of data packets in each buffer in the network routers, respectively.
#
# In the service center we might be interested in the average waiting time during a day. In the warehouse we might be interested in the long-run probability of having no stock. In the case of the progression of a disease we might be interested in the time until death, while in the network situation we might be interested in the percentage of packets lost due to buffer overflow.
#
# When employing DES there are two very distinct options: you can program the simulation in a programming language, or you can use a (graphical) tool. For certain programming languages there are libraries available with useful entities for simulation, but most code has to be programmed. Graphical simulation tools require less programming. For an impression of how such a tool works see, Figure 5.1. You can drag and drop components at the left to make a model in the middle. By clicking on the components they can be configured. A large number of graphical simulation tools exist, mostly proprietary, some of them focused on specific applications. The advantages of both methods are clear: programming offers flexibility and computational speed; using a tool offers speed in implementation plus a configurable graphical interface to impress customers.

# %% [markdown]
# ![Figure 5.1: An impression of a simple model in the Arena DES tool](images/lecture12_fig5.1.png)

# %% [markdown]
# > **Box 5.4. Object-oriented programming**
# >
# > Simulation lends itself perfectly to object-oriented (OO) programming, which is the paradigm behind many modern programming languages such as Java, C++ and C#. In fact, the simulation language Simula, developed in the 1960s, is generally considered to be the first OO programming language. It had a considerable influence on current-day OO languages.
#
# **Exercise 5.4** Consider a small intensive care units with 2 beds. Patients arrive with exponentially distributed interarrival times, on average every 8 hours. Patients stay for a lognormal duration with parameters 1 and 1. When both beds are occupied patients are transferred to a different hospital. Simulate this ICU for one week and count the number of transfers. Do this for multiple runs and construct a CI.
#
# This exercise requires programming experience. It is useful to store and update at the time of each event the current time, the number of occupied beds, and times of the next arrival and departures. There is a Wikipedia page with details of the exponential distribution.
#
# **Validation**
#
# Statistics and ML also play an important role in DES, but in a different way than when you apply them directly. In simulation the system under study is considered to be composed of components, who interact in a known way. However, to derive the parameters of the components and their interaction we need statistics and ML. Validation is concerned with the question to which extend the simulation reflects reality, i.e., to which extent errors in the components and their interaction propagate to the level of the performance measures. Therefore, the parameters of the components are determined using appropriate ML techniques, and then the performance of the simulation is compared to that of the real system. An accurate approximation of the components does not guarantee an accurate output: (simulation) models can be more or less robust to parameter errors.
#
# Validation is rarely easy, often because of a lack of data. Quite often the differences between reality and simulation are so big that statistical tests for equality are always rejected. This does not mean that simulation is useless. In evaluating the differences, we should always take the goal of the simulation into account. Still, in many situations simulations can only be validated after considerable effort or even tuning (where certain parameters are adapted to make the simulation model fit reality). Simulation should only be used after careful consideration: implementing simulation is time-consuming and there is no guarantee of reliable results.
#
# **Example 5.4** Consider an emergency department (ED) of a hospital. Every ED has limited care facilities, often leading to congestion and delays. Four hours is generally considered to be the limit to the length of stay of patients at an ED. Our ED wants to analyze the factors that lead to higher numbers of patients staying longer than 4 hours.
#
# Simulation requires many resources within the ED to be modeled: triage nurses, doctors with different specialties, beds, radiology equipment, etc. For all these resources, parameters such as their duration have to be determined, as well as the routing between the resources, priority of treatment, etc. Reliable data is hard to get, among other reasons because data entry is of little importance and because of the omnipresence of ad-hoc decisions. This lack of reliable data often translates into output that is far from reality. This makes simulation of limited use to systems where human decision making plays such a central role.
#
# We could use an ML approach directly. We should engineer our features as to obtain parameters of interest. Then we could train our model and determine the impact of, for example, high numbers of arrivals or lateness of doctors on the length of stay at the ED. There is no such thing as validation in ML; the risk here is overfitting.
#
# An ML approach is much faster than simulation and often gives very good results. In certain situations, we cannot avoid the use of simulations, especially when we are interested in situations where we do not have data. For example, suppose we want to change the routing in the ED. Then all components are analyzed using historical data. The model is validated on the current way of working, and then the new situation is simulated.
#
# > **Box 5.5. Long-run performance**
# >
# > Sometimes there is no natural termination moment for the simulation. In the supermarket example a day might be the right time frame, but in the network simulation there might not be such a moment. We are interested in the long-run stationary performance for constant parameters. Under certain conditions it can be shown (using the LLN) mathematically that the long-run average performance approaches the long-run expected performance. Because we cannot simulate for an infinitely long period, and because a single run does not give us information on the variability, it is customary to take the average over a number of runs. To avoid different "start-up" behavior, the first part of each simulation is not counted. See the figure below for an illustration of a service center with 10 counters. We clearly see the average over 100 runs increasing from the empty initial situation to around 15, and two runs constantly fluctuating.

# %% [markdown]
# ![Box 5.5: number of customers over time for a service center with 10 counters — the average over 100 runs rises from empty to around 15, while two individual runs fluctuate constantly](images/lecture12_box5.5.png)

# %% [markdown]
# #### 5.3 Additional reading
#
# There are many books on the mathematical aspects of simulation. See, e.g., Ross [33]. Kelton et al. [20] is an example of a book which is more focused on modeling and tooling (especially the discrete-event simulation tool Arena). There is a list of discrete-event software tools on Wikipedia.
#
# Savage [36] uses simulation to explain variability and its pitfalls to layman, avoiding words such as random variable. Klastorin [21] is an excellent book on project management.
#
# More information on the OO simulation language Simula can be found on Wikipedia.

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Simulation (Ch. 5); "Read Chapter 3 (as a recap)" (listed under "Before lecture" in the schedule)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — Chapter 3 "Variability" (whole, recap); Chapter 5 "Simulation" (whole)
# - **Errata applied:** none
#
# > Chapter 3 is included as recap material (per the schedule's "Before lecture" prep reading), placed before the Chapter 5 lecture content.
