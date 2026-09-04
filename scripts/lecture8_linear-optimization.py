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
# # Lecture 8: Linear Optimization

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 6 — Linear Optimization (§6.1-§6.3)
#
# In this chapter we discuss linear optimization problems. It is a framework used successfully in many industries to solve a broad variety of problems. We discuss different types of linear problems and show how you can solve them in Excel and R.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - implement linear optimization problems in R, Excel and dedicated modeling languages
# - model appropriate business problems as linear optimization models
# - reflect on the usefulness and applicability of linear optimization

# %% [markdown]
# #### 6.1 Problem formulation
#
# We introduce linear optimization through an example. Later on we will give the general formulation.
#
# Assume a company has $n$ products which it can produce using $m$ resources of which there is only a limited amount available. The problem to solve is which quantities to produce of each product such that the revenue is maximized and the resource constraints are satisfied. Examples of this *product-mix problem* are refineries combining different types of crude oil into end products, a farmer dividing his land between crops with constraints on the amount of fertilizer or environmental impact, etc.
#
# Let us consider a simple instance of this problem. A company has 2 differents products, for example 2 types of crops, with profit 2 and 3 per quantity produced. We also have 2 resources: fertilizer and land. Product 1 requires 1 unit (e.g., ton) of fertilizer and 1 unit (e.g., acre) of land per unit produced, product 2 requires only 2 units of land. The availability of the resources is as follows: 5 units of fertilizer, 10 units of land. What is the optimal product mix?
#
# This problem has a structure in which we can apply to many optimization problems:
#
# - there are decision variables that have to be chosen;
# - there is an objective that needs to be maximized;
# - there are constraints which need to be satisfied.
#
# In mathematical terms, our instance becomes:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x_1 + 3x_2 & \text{(objective)} \\
# \text{subject to} \quad & x_1 \le 5 & \text{(constraint resource 1)} \\
# & x_1 + 2x_2 \le 10 & \text{(constraint resource 2)} \\
# & x_1, x_2 \ge 0.
# \end{aligned}
# $$
#
# Because the functions $2x_1 + 3x_2$, $x_1$ and $x_1 + 2x_2$ are linear in $x = (x_1, x_2)$ we call this a linear optimization (LO) problem. For LO, efficient solvers exist that are guaranteed to give an optimal solution, even for problems with thousands of variables and constraints. To solve our instance with R, the following code can be used:
#
# ```r
# > install.packages("lpSolve")                          # install solver package (use only once)
# > library(lpSolve)                                      # load library
# > f.obj <- c(2, 3)                                       # set objective
# > f.con <- matrix (c(1, 0, 1, 2), nrow=2, byrow=TRUE)     # constraint values
# > f.dir <- c("<=", "<=")                                  # constraint types
# > f.rhs <- c(5, 10)                                       # resource amounts
# > lp ("max", f.obj, f.con, f.dir, f.rhs)                  # get optimal value
# > lp ("max", f.obj, f.con, f.dir, f.rhs)$solution         # get optimal solution
# ```
#
# **Exercise 6.1** Solve the problem of Section 6.2 with R.
#
# There are different ways to understand what the R solver did. Let us first take a graphical look. In Figure 6.1 the problem is drawn with $x_1$ on the horizontal axis and $x_2$ on the vertical one. We see the two constraints who together with the non-negativity constraints ($x_1, x_2 \ge 0$, assumed by default) delimit the allowable area, often called the feasible region. Because of the linearity of the constraints and the objective, the optimum (if it exists, see below) must be at the edge, at a corner. To determine the optimal corner, we slide a line with equal objective value until we hit the feasible region, i.e., the set of points that satisfy all constraints. The line with value 36 is drawn in the figure. When we slide it down it hits the feasible region in the point with $x_1 = 5$ and $x_1 + 2x_2 = 10$. From this, the optimal solution follows again: $x_1 = 5$ and $x_2 = 2.5$.

# %% [markdown]
# ![Figure 6.1: A graphical view of LO](images/lecture8_fig6.1.png)
#
# > **Erratum applied (p. 87):** the objective line actually drawn in Figure 6.1 corresponds to value 24, not 36 (the text above refers to the line "with value 36").

# %% [markdown]
# Let us now take an algebraic point of view. The 2 constraints can be rewritten as equalities as follows, using additional variables $y_1$ and $y_2$:
#
# $$
# \begin{aligned}
# x_1 \le 5 \\
# x_1 + 2x_2 \le 10 \\
# x_1, x_2 \ge 0
# \end{aligned}
# \quad\Leftrightarrow\quad
# \begin{aligned}
# x_1 + y_1 &= 5 \\
# x_1 + 2x_2 + y_2 &= 10 \\
# x_1, x_2, y_1, y_2 &\ge 0
# \end{aligned}
# $$
#
# We have 2 equalities and 4 variables. This means that 2 non-zero variables suffice to find a solution. All 4 corners of the feasible region correspond to such a solution. The interior corresponds to solutions for which all variables are positive. The corners correspond to the following solutions:
#
# - $(x_1, x_2, y_1, y_2) = (0, 0, 5, 10)$, origin;
# - $(x_1, x_2, y_1, y_2) = (0, 5, 5, 0)$, upper-left corner;
# - $(x_1, x_2, y_1, y_2) = (5, 0, 0, 5)$, lower-right corner;
# - $(x_1, x_2, y_1, y_2) = (5, 2.5, 0, 0)$, upper-right corner (the optimum).
#
# The algorithm implemented in the solver hops from corner to corner until it cannot improve the objective value anymore. This method is called the simplex algorithm.
#
# > **Box 6.1. History of Linear Optimization**
# >
# > Several researchers have formulated Linear Optimization problems but it was G.B. Dantzig (1914-2005) who invented in 1947 the simplex algorithm. Dantzig was an American scientist with German-French roots. At the time, until very recently, it was commonly known as linear programming. LO has been extremely useful for solving all kinds of business problems and is by far the most successful technique within operations research.
# >
# > Its random, dynamic counterpart is called dynamic programming (see Chapter 9), developed by R.E. Bellman (1920-1984). This division between deterministic and random problems is still very visible in operations research theory and applications.
#
# The general formulation of an LO problem is as follows, for $n$ decision variables and $m$ constraints:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{i=1}^{n} p_i x_i & \text{(objective)} \\
# \text{subject to} \quad & \sum_{j=1}^{n} a_{ij} x_j \le b_i, \quad i = 1, \dots, m & \text{(constraints)} \\
# & x_1, \dots, x_n \ge 0.
# \end{aligned}
# $$
#
# Instances where the objective needs to be minimized or with constraints of the form "=" or "≥" can be rewritten to fit the general formulation.
#
# **Exercise 6.2** Consider the following LO problem:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & 2x_1 + x_2 + 4x_3 & \text{(objective)} \\
# \text{subject to} \quad & x_1 - 2x_2 + 2x_3 \le 120 \\
# & -x_1 - x_2 + 3x_3 = 100 \\
# & x_1 - x_2 + x_3 \ge 80 \\
# & x_i \ge 0 \text{ for all } i.
# \end{aligned}
# $$
#
# a. Solve it in R (see the online documentation of the R package lpSolve to find out how to change the signs of the constraints).
#
# b. Rewrite it in the general form with maximization and "≤" constraints.
#
# c. Solve this problem in R.
#
# Sometimes we write the general formulation in matrix notation. Then it becomes:
#
# $$
# \max\{p^T x \mid Ax \le b,\ x \ge 0\},
# $$
#
# where $p$ and $x$ are $n$-dimensional column vectors (and thus $p$ transposed, $p^T$, is a row vector), $b$ is an $m$-dimensional column vector, and $A$ is an $m \times n$ matrix.
#
# Not all LO problems can be solved. Sometimes the problem is unbounded, meaning that solutions of arbitrarily large values can be found. On the other hand, there are also problems where there are no feasible solutions at all. In Figure 6.2 examples of both situations are shown.

# %% [markdown]
# ![Figure 6.2: An unbounded (left) and an infeasible (right) LO problem](images/lecture8_fig6.2.png)

# %% [markdown]
# **Exercise 6.3** Enter both problems of Figure 6.2 in R and see what output you get.

# %% [markdown]
# #### 6.2 LO in Excel
#
# To solve LO problems in Excel, you should first make sure that the "solver add-in" is installed. You can check its availability under "Tools". How to install it depends on your version of Excel, for more details do a Google search for "install Excel solver".
#
# Now we show how to solve the following problem using Excel:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x_1 + 4x_2 + 8x_3 & \text{(objective)} \\
# \text{subject to} \quad & x_1 + 3x_2 + 2x_3 \le 10 & \text{(constraint 1)} \\
# & x_1 + 3x_3 \le 12 & \text{(constraint 2)} \\
# & x_1, x_2, x_3 \ge 0.
# \end{aligned}
# $$
#
# The first step is to enter this problem in Excel in such a way that the decision variables, the objective value and the constraint values are in separate cells. See Figure 6.3 for a possible implementation of the above problem. For the decision variables we used arbitrary values (1, 2, 3).

# %% [markdown]
# ![Figure 6.3: LO implementation in Excel](images/lecture8_fig6.3.png)

# %% [markdown]
# The following formulas were entered:
#
# - cell E4: `=SUMPRODUCT(B$2:D$2,B4:D4)`
# - cell E7: `=SUMPRODUCT(B$2:D$2,B7:D7)`
# - cell E8: `=SUMPRODUCT(B$2:D$2,B8:D8)`
#
# Thanks to the $-signs, the formula only has to be entered once and can then be copied.
#
# We are now ready to open the solver dialog. After entering the right values the dialog should look like Figure 6.4. The dialog can look slightly different, depending on your version of Excel. Hitting "Solve" will now solve the problem to optimality, with objective value 34.67.
#
# **Exercise 6.4** Solve the problem of Exercise 6.2 using Excel. Note that the constraints can be entered one by one each having a different sign.
#
# **Exercise 6.5** The tax office can only check a subset of the tax declarations it received. There are 3 types of employees with different skills, and 3 types of declarations. Per declaration type, the expected revenues from additional taxation are as follows: (200, 1000, 500) Euros. Every tax employee can process every declaration, except for employee type 2 who cannot process declaration type 2 and employee type 3 who cannot process declaration type 3. The time per declaration depends on the declaration type and is (1, 3, 2) hours, respectively, except for employee type 3 who takes 2 hours for a type 1 declaration. The numbers of declarations are (15000, 6000, 8000), the numbers of available hours are (10000, 20000, 15000). How do you assign the employees to the different declaration types? Use Excel to solve this problem.

# %% [markdown]
# ![Figure 6.4: Excel solver dialog](images/lecture8_fig6.4.png)

# %% [markdown]
# The standard Excel solver is rather limited in capabilities. A simple alternative is the "OpenSolver" which is easy to install and comparable in use, but comes with a stronger solver without limitations on the numbers of variables or constraints. See OpenSolver.org.
#
# **Exercise 6.6** Extend the problem of Figure 6.3 in the following ways. When solving these problems it helps to think what the additional decision is that needs to be taken.
#
# a. Assume that, next to the 10 units available, you can buy extra units of resource 2 for the price of 1 per unit. What is the optimal solution now?
#
# b. The same question, but now you can buy resource 1 for 2 per unit.
#
# c. The same question, but now you can buy resource 1 for 1 per unit. Can you interpret the result?

# %% [markdown]
# #### 6.3 Example LO problems
#
# In this section we consider several types of optimization problems that can be solved using LO.
#
# A graph is the mathematical name for a network consisting of nodes and edges connecting the nodes. Nodes are sometimes also called vertices. Edges can be directed or undirected, i.e., uni-directional or bi-directional. Directed edges are often called arcs. Many practical problems can be formulated as problems on graphs. One such problem is project planning.
#
# A project consists of a number of activities, each having a duration. These are the nodes in the graph. Additionally, certain activities require others to finish before they can get started. These precedence relations are modeled as directed edges in the graph. In Figure 6.5 an example of such a graph is given. We need to determine the earliest finish time of each activity. These are the decision variables, $x_i$ for activity $i$. Every precedence relation leads to a constraint. When $i$ precedes $j$ then this can be enforced by the constraint $x_i + d_j \le x_j$, where $d_j$ is the duration of activity $j$. For example, in the project of Figure 6.5 we model the relation $F \to G$ by the constraint $x_F + 2 \le x_G$.

# %% [markdown]
# ![Figure 6.5: Directed graph with weights of a project planning problem](images/lecture8_fig6.5.png)

# %% [markdown]
# We are interested in the time at which all activities are finished. This can be modeled by an additional variable $z$, bigger than all finish times, that has to be minimized. This leads to the following LO formulation:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & z \\
# \text{subject to} \quad & z \ge x_i \text{ for all vertices } i \\
# & x_i + d_j \le x_j \text{ if } i \text{ precedes } j \\
# & x_i \ge d_i \text{ for all vertices } i.
# \end{aligned}
# $$
#
# The last constraint ensures that no activity starts before time 0. Note that the finish time can also be found using an algorithm without LO, and that this algorithm can be extended to random activity durations. This is relevant in practice because often durations are hard to predict accurately, which is one of the main reasons why, for example, IT projects often finish after the scheduled deadline.
#
# **Exercise 6.7** Formulate the project planning problem of Figure 6.5 as LO problem and solve it using Excel.
#
# A more complicated problem that can be solved using LO is the so-called transportation problem. In this problem we have to transport a single type of good from $n$ sources to $m$ destinations. Source $i$ has supply $a_i$, destination $j$ has demand $b_j$, and link $i \to j$ has transportation costs $c_{ij}$ per unit transported on it. The question is: For each link, how much should you ship on it in order to satisfy the demand of each destination without violating supply constraints? See Figure 6.6 for an illustration. When link $i \to j$ does not exist we can take $c_{ij} = \infty$.
#
# The LO formulation is as follows:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{j=1}^{m} x_{ij} \le a_i \text{ for } i = 1, \dots, n; \\
# & \sum_{i=1}^{n} x_{ij} \ge b_j \text{ for } j = 1, \dots, m; \\
# & x_{ij} \ge 0 \text{ for all } i, j.
# \end{aligned}
# $$
#
# **Exercise 6.8** Solve the following transportation problem, where "x" means there is no connection.

# %% [markdown]
# ![Figure 6.6: Transportation problem — supply $a_i$ per source $i$, demand $b_j$ per destination $j$, and costs $c_{ij}$ per source-destination pair (source 1: $a_1=10$, costs to destinations 1-4 are 0, 5, x, 0; source 2: $a_2=6$, costs 4, 6, 4, 3; source 3: $a_3=10$, costs 2, 4, 4, 6; demands $b_j$: 5, 5, 5, 5)](images/lecture8_fig6.6.png)

# %% [markdown]
# If we add intermediate nodes to the transportation problem, as in Figure 6.7, then we obtain the transshipment problem. We can solve it by adding constraints of the form:
#
# $$
# \sum_{i=1}^{n} x_{ik} = \sum_{j=1}^{m} x_{kj}
# $$
#
# for all intermediate nodes $k$. It can be extended to a network by adding multiple layers of intermediate nodes.
#
# Next we consider multi-period production/inventory models. Here we assume there are multiple time periods, say $t = 1, \dots, T$, and a starting inventory $s_0$. Every day, the amount of production or supply has to be decided: $x_t$. There is demand, $d_t$ at time $t$, holding costs $h_t$, and production costs $c_t$. The problem is to find the production/order schedule that minimizes the total costs. This can be formulated as an LO problem as follows:

# %% [markdown]
# ![Figure 6.7: Transshipment problem](images/lecture8_fig6.7.png)

# %% [markdown]
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{t=1}^{T} (c_t x_t + h_t s_t) \\
# \text{subject to} \quad & s_{t+1} = s_t - d_{t+1} + x_{t+1} \text{ for } t = 1, \dots, T-1; \\
# & x_t, s_t \ge 0 \text{ for } t = 1, \dots, T.
# \end{aligned}
# $$
#
# This model can be extended in many different directions, such as maximum stock or production capacity, multiple products and resources, backorders, and fixed order costs (see Exercise 6.19).

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Linear optimization (§6.1-§6.3); "Modeling" (see [Lecture 8: Introduction to Business Analytics](lecture8_introduction.ipynb) for the uncertainty note on this schedule topic)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — §6.1 "Problem formulation"; §6.2 "LO in Excel"; §6.3 "Example LO problems"
# - **Errata applied** (per `Literature/Erratum Book An Introduction to Business Analytics by Koole (2019).pdf`)**:**
#   - p. 87 (§6.1, Figure 6.1): the objective line drawn in the figure has value 24, not 36
