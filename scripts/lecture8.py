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
# Lecture 8: Introduction to Business Analytics; Modeling; Linear and Integer Optimization

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Introduction to business analytics (Ch. 1); Modeling; Linear optimization (§6.1-§6.3); Integer linear optimization (§6.4)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — Chapter 1 "Introduction" (whole); §6.1 "Problem formulation"; §6.2 "LO in Excel"; §6.3 "Example LO problems"; §6.4 "Integer problems"
# - **Errata applied:** none
#
# > **Uncertainty:** the schedule lists a separate topic "Modeling" between Ch. 1 and §6.1-§6.3, with no section number and no dedicated book section. It is not mapped to a specific passage below to avoid guessing.

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 1 — Introduction
#
# This chapter explains business analytics and data science without going into any technical detail. We will clarify the meaning of different terms used, put the current developments in a historical perspective, give the reader an idea of the potential of business analytics (BA), and give a high-level overview of the steps and pitfalls in implementing a BA strategy.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - describe in non-technical terms the field of business analytics, the different steps involved, the connections to other fields of study and its historical context
# - reflect on the skills and knowledge required to successfully apply business analytics in practice

# %% [markdown]
# #### 1.1 What is business analytics?
#
# According to Wikipedia, "Business analytics refers to the skills, technologies, practices for continuous iterative exploration and investigation of past business performance to gain insight and drive business planning." In short, BA is a rational, fact-based approach to decision making. These facts come from data, therefore BA is about the science and the skills to turn data into decisions. The science is mostly statistics, artificial intelligence (data mining and machine learning), and optimization; the skills are computer skills, communication skills, project and change management, etc.
#
# It should be clear that BA by itself is not a science. It is the total set of knowledge that is required to solve business problems in a rational way. To be a successful business analyst, experience in BA projects and knowledge of the business areas that the data comes from (such as healthcare, advertising, finance) is also very valuable.
#
# BA is often subdivided into three consecutive activities: descriptive analytics, predictive analytics, and prescriptive analytics. During the descriptive phase, data is analyzed and patterns are found. The insights are consequently used in the predictive phase to predict what is likely to happen in the future, if the situation remains the same. Finally, in the prescriptive phase, alternative decisions are determined that change the situation and which will lead to desirable outcomes.
#
# **Example 1.1** A hotel chain analyzes its reservations to look for patterns: which are the busiest days of the week? What is the impact of events in the city? Is there a seasonal pattern? Etc. The outcomes are used to make a prediction for the revenue in the upcoming months. By changing the pricing of the rooms in certain situations (such as sports events), the expected revenue can be maximized.
#
# Analytics can only start when there is data. Certain organizations already have a centralized data warehouse in which relevant current and historical data is stored for the purpose of reporting and analytics. Setting up such a data warehouse and maintaining it is part of the business intelligence (BI) strategy of a company. However, not all companies have such a centralized database, and even when it exists it rarely contains all the information required for a certain analysis. Therefore, data often needs to be collected, cleansed and combined with other sources. Data collection, cleansing and further pre-processing is usually a very time-consuming task, often taking more time than the actual analysis.
#
# **Example 1.2** In the hotel revenue management example above we need historical data on reservations but also data on historical and future events in the surroundings of the hotel. There are many reasons why this data can be hard to get: reservation data may only be stored at an aggregated level, there may have been changes in IT systems which overrode previously collected data, there may be no centrally available list with events, etc. Many organizations assume they already have all the data required, but as soon as the data scientist asks for reservation data combined with the date the booking was made or the event list from the surrounding area, the hotel might find out that they lack data.
#
# Therefore, data collection and pre-processing are always the first steps of a BA project. Following the data collection and pre-processing the real data science steps begin with descriptive analytics. Moreover, a BA project does not end with prescriptive analytics, i.e., with generating an (optimal) decision. The decision has to be implemented, which requires various skills, such as knowledge of change management.
#
# To summarize, we distinguish the following steps in a BA project:

# %% [markdown]
# ![Data collection, data pre-processing, descriptive analytics, predictive analytics, prescriptive analytics, implementation — the full BA project spans all steps, the data science steps span descriptive through prescriptive analytics.](images/lecture8_fig-steps.png)

# %% [markdown]
# The model above suggests a linear process, but in practice this is rarely the case. At many of the steps, depending on the outcome, you might revisit earlier steps. For example, if the predictions are not accurate enough for a particular aaplication then you might collect extra data to improve them. Furthermore, not all BA projects include prescriptive analytics, many projects have insight or prediction as goal and therefore finish after the descriptive or predictive steps.
#
# The major scientific fields of study corresponding to these BA steps are:

# %% [markdown]
# ![Business intelligence, cleaning & feature engineering, data analysis & visualisation, statistics & machine learning, optimization & reinforcement learning, soft skills.](images/lecture8_fig-fields.png)

# %% [markdown]
# Next to cleansing, feature engineering is an important part of data preparation, to be discussed later. During descriptive analytics you get an understanding of the data. You visualize the data and you summarize it using the tool of statistical data analysis. Getting a good understanding is crucial for making the right choices in the consecutive steps.
#
# Following the descriptive analytics a BA project continues with predictive analytics. A target value is specified which we want to predict. Based on the data available, the parameters of the selected predictive method are determined. We say that the model is trained on the data. The methods originate from inferential statistics and machine learning, which have their respective roots in mathematics and computer science. Although the approach and the background of these fields are quite different, the techniques largely overlap.
#
# **Example 1.3** A debt collection agency wants to use its resources, mainly calls to debtors, in a better way. It collects data on payments which is enriched by external data on household composition and neighborhood characteristics. After the data analysis and visualization a method is selected that predicts, given the characteristics of the dept and the actions taken by the agengy, the probability that the deptor will pay off their debt. In the prescriptive step, which is to be discussed next, the best action for each deptor is determined.
#
# Finally, during the prescriptive analytics phase, options are found to maximize a certain objective. Because the future is always unpredictable to a certain extent, optimization techniques often have to account for this randomness. The field that specializes in this is (mathematical) optimization. It overlaps partially with reinforcement learning, which has its roots in computer science. A special feature of reinforcement learning is that prediction and optimization are integrated: it combines in one method the predictive and prescriptive phases.
#
# **Example 1.4** Consider again Example 1.2 on hotel revenue management. After having studied the influence of events and for example intra-week fluctuations on hotel reservations in the descriptive step demand per price class is forecasted in the predictive step. These forecasts are input to an optimization algorithm that determines on a daily basis the prices that maximize total revenue.
#
# We end this section by discussing two terms that are closely related to BA: Data science and big data. Data science is an older term which has recently shifted in meaning and increased in popularity. It is a combination of different scientific fields all concerned with extracting knowledge from data, mainly data mining and statistics. Part of the popularity probably stems from the fact that the Harvard Business Review called a data scientist role "the sexiest job of 21st century", anticipating the huge demand for data scientists. The knowledge base of data scientists and business analysts largely overlap. However, the deliverable of BA is improved business performance, whereas data scientists focus more on methods and insights from data. Improved business performance requires optimization to generate decisions and soft skills to implement the decisions.
#
# Finally, a few words on big data. Big data differentiates itself from regular data sets by the so-called 3 V's: volume, variety, and velocity. A data set is considered to be "big data" when the amount of data is too much to be stored in a regular database, when it lacks a homogeneous structure (i.e., free text instead of well-described fields), and/or when it is only available real-time. Big data requires adapted storage systems and analysis techniques in order to exploit it.
#
# > **Box 1.1. From randomized trials to using already available data**
# >
# > The traditional way to do scientific research in the medical and behavorial sciences is through (double-blind) randomized trials. This means that subjects (e.g., patients) have to be selected, and by a randomized procedure they are made part of the trial or part of the control group. It is called double blind when the subject and the researcher are both not aware of who is in which group. This kind of research set-up allows for a relatively simple statistical analysis, but it is often hard to implement and very time-consuming.
# >
# > Nowadays, data can often be obtained from Electronic Health Records and other data sources. This eliminates the need for separate trials. However, there will be all kinds of statistical biases in the data, making it harder to make a fair comparison between treatments. For example, patients of a certain age or having certain symptoms might get more-often a certain treatment. This calls for advanced statistical methods to eliminate these biases. These methods are usually not taught in medical curricula, requiring the help of expert data scientists.
#
# Big data now receives a lot of attention due to the speed at which data is collected these days. As more and more devices and sensors automatically generating data are connected to the internet (the internet of things) again, the amount of stored data doubles approximately every 3 years. However, most BA projects do not involve big data, but use with relatively small and structured data sets. It might have been the case that such a dataset had its origin in big data from which relevant information has been extracted.
#
# **Example 1.5** Cameras in metro stations are used to surveil passengers. Using image recognition software the numbers of passengers can be extracted, which can be used as input for a prediction method that forecasts future passenger volumes.

# %% [markdown]
# #### 1.2 Historical overview
#
# Business analytics combines techniques from different fields all originating from their own academic background. We will touch upon the main constituent fields of statistics, artificial intelligence, operations research, and also BA and data science (DS).
#
# Statistics is a mathematical discipline with a large body of knowledge developed in the pre-computer age. For many decades, statistics has been taught at universities without the use of any data sets. The central body of knowledge concerns the behavior of statistical quantities in limiting situations, for example when the number of observations approaches infinity. This is of a highly mathematical nature. More recently new branches of statistics have come into existence, many of which are more experimental in nature. However, quite often statistics is still taught as a mathematical discipline with a focus on the mathematics.
#
# Artificial intelligence (AI) is a field within computer science that grew rapidly from the 1970s with the advent of computers. Initial expectations were highly inflated. One believed, for example, that so-called expert systems would soon replace doctors in their work of diagnosing illnesses in patients. This did not happen and the attention for AI diminished. Today, the expectations are high again, largely due the fields of data mining and machine learning which are relevant for BA. They developed more recently when large data sets became available for analysis. Both fields of data mining and machine learning focus on learning from data and making predictions using what is learned. Machine learning focuses on predictive models, data mining more broadly on the process from data pre-processing to predictive analytics, with a focus on data-driven methods. The difference between statistics and machine learning are their origins and the more data-oriented approach of ML: Mathematicians want to prove theoretically that things work, computer scientists want to show it using data.
#
# Operations research (OR) is about the application of mathematical optimization to decision problems in organizations. OR, sometimes called management science, and abbreviated as OR/MS, also raised big expectations, in the 1950s, following the first successes of the allied forces of OR being applied during World War II. The belief was that scientific methods would replace traditional management and turn it into a science. However, the impact at the strategic decision level remained very limited and OR applications are mainly found at the operational level. Quite often the application of OR would be to a logistical problem such as the routing of delivery vans, outside the scope of higher management. OR faces the same problems as statistics: it has been developed as a highly mathematical science, but it has a hard time adapting itself to the current situation in which data and tooling is easily available. Often it is still taught in a highly abstract mathematical way, limiting the potential impact in practice.
#
# BA on the other hand, developed in organizations that realized that their data was not just valuable for their current operations, but also to gain insight and improve their processes. Starting in the 1990's, we saw more and more analysts working with data in organizations. An important difference with OR is that many executives do understand the value of analytics and adopt a company-wide BA strategy. A book by Davenport [8], who is an advocate of BA, also played a role in increasing the interest in the value of analytics to executives. Interestingly enough, the main example throughout the book is dynamic pricing in airlines, a typical OR success. The name OR is not mentioned once. This supports the opinion that some of these new areas are in fact rebranded old areas, it's old wine in a new bottle. Whether this is really true, or if there are fundamental differences between areas is not really relevant. The fact is that the availability of data, computers and software made the widespread use of BA possible. Finally, BA methods — also the ones originating from the mathematical sciences — are used on a huge scale in companies, institutions and research centers, offering countless opportunities for business analysts and data scientists.
#
# DS as a term has been around for a long time. In the end of the last century it was mainly associated with statistics. Much like BA, the term became popular with the availability of large data sets. However, today it is more often associated with techniques from computer science such as machine learning. In contrast, BA is more often associated with mathematics and industrial engineering.

# %% [markdown]
# #### 1.3 Non-technical overview
#
# In this section we give a non-technical overview of the most often used techniques and explain some of the technical terms that are regularly used. This section by nature can only be an oversimplification of reality, but it will help to get a flavor of the totality of the field, which even professionals in the field sometimes do not have. The techniques we discuss in this section are summarized in Figure 1.1.
#
# The four steps pre-processing, descriptive, predictive and prescriptive analytics, can also be described as follows:
#
# - preparing the data set;
# - understanding the data set;
# - predicting a target value;
# - maximizing the target value.

# %% [markdown]
# ![Figure 1.1: An overview of the most-often used data science techniques](images/lecture8_fig1.1.png)

# %% [markdown]
# Most predictive techniques require that you first structure the data. For example, topics can be extracted from text entered on social media or types of objects can be extracted from images. This brings us to a first distinction: between structured and unstructured data. Structured data usually consists of entries (e.g., people) with attributes (e.g., name, income, sex, nationality). The possible value for the attributes are well-defined (e.g., numerical, M/F, standard country codes). Structured data can be represented as a matrix: the rows are the entries, the columns the attributes.
#
# Structured data comes in different flavors: for example, it can be numerical (e.g., temperature), categorical (e.g., days of the week), binary (e.g., true/false). Depending on the type of data different algorithms or adaptations of algorithms are used. If we have univariate data, i.e., data with only one attribute, then we can look at the distribution or compare different data sets. For multivariate data we can study how the different attributes influence each other.
#
# Unstructured data has no such structure. It might be data from cameras, social-media sites, text entered in free text fields, etc. Counted in bytes, unstructured data is the majority of the data that is stored today, and it is often also big data. However, most of the BA and DS projects involve structured data, on which we will focus. When working with unstructured data, the first step is often to extract features to make it structured and therefore suitable as input for an algorithm working with structured data (e.g., images from road-side cameras are used to extract license plates which are then used to analyze the movement of cars).
#
# Dealing with unstructured data is an important part of the pre-processing step. Cleansing is another one. Data often contains impossible values or empty fields. Different techniques exist to deal with these. A final important pre-processing activity is feature engineering, combining attributes or features into new potentially more useful attributes. For example, combining "day of week" and "time" can lead to an attribute "business hours", and postal codes of individuals combined with census data can lead to an approximation of income and family composition.
#
# Next we explore the data in the descriptive step. Typical activities are visualisation, different statistical techniques such as hypothesis testing, and clustering. Visualization is a technique as old as humanity, but it has developed tremendously over the last decades. Exploratory statistics will be discussed in Chapter 3. In clustering, you look for data points that are in some mathematical sense close together. Think about clustering individuals based in income, sex, age and family composition for marketing purposes.
#
# In the descriptive step we do not focus on a target value (such as sales or number of patients cured). Having a target value is the defining distinction of predictive analytics. Therefore predictive analytics is also called supervised learning: we learn an algorithm to predict a target value based on a data set with known target values. In contrast, techniques such as clustering are considered unsupervised learning.
#
# Supervised learning comes in two flavors: regression and classification. In regression we estimate a numerical value. The best-known methods are linear regression and artificial neural networks (which is actually a form of non-linear regression), but other methods exist. In classification, the outcome is membership of two or more classes, e.g., whether or not somebody will click on an online ad, or vote on one of a number of parties. Most methods for regression can be adapted such that they can classify as well. Machine learning covers both supervised and unsupervised learning.
#
# > **Box 1.2. Human versus artificial intelligence**
# >
# > Certain AI techniques are inspired by human intelligence or structures we find in nature, illustrated by names such as artificial neural networks or evolutionary computing. It is an interesting question whether or not we should try to copy human behavior with, eventually, the possibility that computers become "more intelligent" than humans. We could also argue that humans and computers have different capacities (seeing structures versus fast and errorless computation) and that our approaches to solving the same problem should be completely different. Your point of view might influences whether or not you find AI dangerous, as Stephen Hawkins did for example.
#
# Often the set of known data entries is split in a training and a test set: the algorithm is trained on the basis of the training set, and then evaluated on the basis of the test set. Usually an algorithm performs worse on the test set, but this is a more reliable comparison, as it avoids overfitting: the fact that the prediction of the algorithm is perfect for the training set but has no predictive value and therefore works bad on the test set. In statistics the terms in sample and out of sample are used for the same concepts. Understanding the background of the techniques and learning how to use them in the data science tool R is one of the main objectives of this book.
#
# Descriptive analytics is deductive in nature: from the data set, we derive characterizing quantities such as means and correlations. Extending the knowledge from the training data to the whole population is induction. This is what we do in statistics and machine learning as part of predictive analytics. Certain predictive models combine deduction and induction: A real-life system is modeled using components. By predicting the behavior of the components (induction) we can deduce the behavior of the whole system. For example, in this way a production plant or the progression of a disease in a body can be simulated. By changing (the behavior of) certain components different scenarios can be analysed, leading to optimization, i.e., prescriptive analytics. Optimization comes in different flavors. Linear optimization is a powerful framework, used in many planning problems, such as crew scheduling in airlines and logistics. When problems are dynamic (e.g., they evolve over time, such as managing an investment portfolio), then dynamic programming is the right framework. When dynamic optimization is combined with learning, then we speak of reinforcement learning.

# %% [markdown]
# #### 1.4 Tooling
#
# A multitude of tools exist to assist the data analyst with his or her task. We first make a rough division between ad hoc and routine tasks. For routine tasks, standardized and often automated procedures exists for the process steps, often involving dedicated and sometimes even tailor-made software. For example:
#
# - for data collection data warehouses exist with connections with operational IT systems;
# - for distribution companies decision support systems exist that compute the optimal route of delivery trucks, saving many transit hours and petrol.
#
# We will first go into detail on software for ad-hoc tasks. For ad-hoc tasks there are a number of proprietary and open source tools — R (open source) and MS Excel (proprietary) are among the most popular ones. Both allow the user to efficiently manipulate data, often represented as matrices. Each tool functions in very different ways: R manipulates data in a declarative way, very much like programming languages. Additionally, the interactive environment RStudio allows for an easy manipulation of data, R scripts and figures. Excel is essentially a 2-dimensional worksheet, with the possibility to perform calculations in each cell and to add entities such as figures. Both have many useful functions, for example for statistical calculation. Many libraries exist containing algorithms that can be added to these tools, both open source and proprietary. Users can also add new functions or libraries to both tools. For a screenshot of a simple implementation of linear regression in both R and Excel, see Figures 1.2 and 1.3.

# %% [markdown]
# ![Figure 1.2: A simple implementation of linear regression in Excel](images/lecture8_fig1.2.png)

# %% [markdown]
# Excel is omnipresent and is easy to learn, making it the favorite tool for many people doing relatively easy computational tasks. However, Excel is also known for the errors users make with it. This might be partly due to a lack of appropriate training, but the lack of structure also contributes. R enforces more structure, just like programming languages do. Learning R requires more time but it may well be worth the investment.
#
# As mentioned, Excel and R are both analytic environments with many built-in functions. Engines can be called from these environments to perform certain tasks, such as optimization. These engines can also be proprietary or open source. For example, for linear optimization (see Chapter 6), the best solvers, Gurobi and CPLEX, are proprietary; CBC is an example of an open-source solver.
#
# Many other environments exist, often for specific analytics tasks. Examples are SPSS, often used in social sciences for statistical analysis, and

# %% [markdown]
# ![Figure 1.3: A simple implementation of linear regression in RStudio](images/lecture8_fig1.3.png)

# %% [markdown]
# AIMMS, an optimization environment. A special place is taken by Python. Python is a programming language with libraries containing many functions for data analysis. For this reason it is both used for ad hoc analysis and routine tasks, solving the so-called "two-language problem", the fact that you have to move from say R to a language like Java or C++ once you move from a successful pilot to a production system.
#
# Excel, although very different in functionality, is also often used for routine tasks. It has some functionality for this, such as the possibility to connect to databases and to add user-friendly screens, but it lacks others, such as user management. Although in principle everything can be built within Excel, thanks to the underlying programming language VBA (Visual Basic for Applications), in practice it often leads to slow error-prone systems consisting of a spaghetti of multiple sheets referring to each other.
#
# Concerning software for routine tasks, there is a large variety in possible tooling. A major difference is between off-the-shelf and tailor-made software. In the area of prescriptive analytics decision support systems (DSS) form the main category of off-the-shelf software. This is software built for a specific goal, such as the routing of delivery vans or the pricing of hotel rooms. Next to the analytics algorithms, DSS typically have built-in connections to data sources and allow the user to interact with the software in such a way that input and output of the algorithms can be manipulated.
#
# In the area of data collection and descriptive analytics BI tools exist, such as IBM Cognos, that help the user collect data and execute queries. Recently, many tools are built to store and manipulate big data. Google and Amazon are major players in this area with the open-source database and data manipulation systems Hadoop and Mapreduce (mainly developed by Google) and Amazon Web Services, providing big data cloud storage and computing.
#
# Tailor-made analytics software can be written in many different languages. We already mentioned Python, but popular languages include php, Java, C++ and C#, combined with mySQL (open source) or MS SQL server databases. Note the move of proprietary off-the-shelf tooling to cloud-based solutions, taking away the need for expensive servers at the customer site, and making maintenance and support much easier.
#
# In Figure 1.4 you can find an overview of the tools discussed. In this book we will mainly use R.

# %% [markdown]
# ![Figure 1.4: Types of analytics tools with some examples; o = open source, p = proprietary](images/lecture8_fig1.4.png)

# %% [markdown]
# Note that many interfaces exist between the tools and languages in Figure 1.4. From within Excel and general programming languages databases can be accessed; DSS, spreadsheets and optimization environments call optimization engines, etc. Especially with the open source environments R and Python every imaginable data science project can be done, where python is preferred in the case of big data or applications requiring intensive computation. R and python are quickly gaining popularity: there is an enormous community developing new libraries and offering support through websites such as stackoverflow.com.

# %% [markdown]
# #### 1.5 Implementation
#
# A successful implementation of BA requires the right combination of tools and skills from the BA consultant(s). But more is needed: the organization should have reached the right maturity level to make the implementation possible. Let us consider first the required skills of the specialist.
#
# The core knowledge of any BA specialist is the command of suitable tooling (such as R) and a broad understanding of descriptive, predictive and prescriptive methods. Next to that, a specialist might have management skills (project management, change management, communication skills), programming skills (in for example C++ or Python), or deep knowledge on some of the technical areas, often clustered by the scientific disciplines of statistics, machine learning or optimization. These specialists are considered to be "T-shaped": they have breadth and also depth in a certain area. Sometimes people talk even of "Π-shaped", emphasizing the importance of knowledge of the application domain, the second vertical bar. However, the importance of breadth cannot be underestimated: It is important to be able to use the right method for the problems one encounters. Scientists are still too often specialized in one tool (e.g., a hammer) which they use for all problems they encounter (e.g., to put a screw in the wall).
#
# It is crucial to have good analysts, but an organization should also support the deployment of analytics. The extent to which an organization supports a certain concept is called its maturity with respect to this concept. The maturity is measured using maturity models. Different analytics maturity models have been developed. The more mature an organization, the higher the impact of analytics. We illustrate the concept using the "INFORMS Analytics Maturity Model" [1]. It consists of three sets of questions, concerning the organization, its analytics capability, and its data and infrastructure. On the basis of this a score is calculated. For example, an organization with a central data warehouse and a centralized analytics strategy will score higher than a company lacking these.

# %% [markdown]
# #### 1.6 Additional reading
#
# General information on many subjects can be found on Wikipedia. We already mentioned Davenport & Harris [8], which is still an interesting non-technical book to read on the value of BA.
#
# For more background on errors in Excel see Powell et al. [29] and other papers by the same authors.
#
# > **Box 1.3. Legal and ethical aspects**
# >
# > Although not his or her main focus, a data scientist should be aware of legal and ethical aspects. The legal aspects often start with the data collection: are you allowed to get and analyze the data? Some form of data anonymization can be useful in this process. Current laws (such as the EU GDPR regulation) also limit the amount of time you are allowed to keep data, which contradicts the wish to keep as much data as possible for future analysis.
# >
# > There are many privacy issues that have to do with data, like who has access to data about you, who owns it, and how do you know which data is out there about you? Ethical questions also arise around the use of algorithms. On what basis do algorithms make decisions about for example employment? Algorithms can be discriminating because they were trained to do so by the data. On the other hand, a data science approach can also give solutions, for example by communicating all parameters of a predictive model.
#
# Some interesting ideas on the different profiles of data scientists (on which part of Section 1.5 is based) can be found in Harris et al. [14].
#
# More information on project management can be found in Klastorin [21]. A classic on change management is Kotter [22].
#
# You can try the INFORMS Analytics Maturity Model yourself at [1].
#
# A well-know mathematician and author writing on ethical issues of data science is Cathy O'Neil, see for example her TED talks and [27].

# %% [markdown]
# ### Chapter 6 — Linear Optimization (§6.1-§6.4)
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
# #### 6.4 Integer problems
#
# For the simplex algorithm to be used, it is essential that the objective and all constraints are linear. Many extensions exist to non-linear functions. One important class is where there is, in addition to the linear constraints, constraints requiring one or more of the decision variables to be integer (i.e., taking values in $\{0, 1, 2, \dots\}$) or binary (taking values in $\{0, 1\}$). We call these integer linear optimization (ILO) problems.
#
# Note that binary problems are special cases of integer problems: the constraint $x_i \in \{0, 1\}$ is equivalent to the following 2 constraints: $x_i \in \{0, 1, 2, \dots\}$ and $x_i \le 1$.
#
# Product-mix problems where we have to produce integer numbers of items is a good example. In R we can add an additional argument to the solver call:
#
# ```r
# > lp ("max", f.obj, f.con, f.dir, f.rhs, all.int=TRUE)
# ```
#
# In Excel we have to add additional constraints, as in Figure 6.8.

# %% [markdown]
# ![Figure 6.8: Entering integer and binary constraints in Excel](images/lecture8_fig6.8.png)

# %% [markdown]
# **Exercise 6.9** Solve the integer version of the problem of Section 6.1.
#
# The archetypical binary LO problem is the knapsack problem. You have to make a selection out of a set of items. Each item has a revenue and a weight. The goal is to maximize the total revenue with a constraint on the total weight. Typical applications of the knapsack are logistics problems, for example selecting items which have to be transported in trucks, or so-called cutting problems, which arise, for example, in steel plants where you have to cuts plates in pieces of different sizes.
#
# As an example, consider a problem with total weight capacity 11. The items are as follows:
#
# | | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
# |---|---|---|---|---|---|---|---|
# | revenue | 60 | 60 | 40 | 10 | 20 | 10 | 3 |
# | weight | 3 | 5 | 4 | 1.4 | 3 | 3 | 1 |
#
# Formulated as ILO we get:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 60x_1 + 60x_2 + 40x_3 + 10x_4 + 20x_5 + 10x_6 + 3x_7 \\
# \text{subject to} \quad & 3x_1 + 5x_2 + 4x_3 + 1.4x_4 + 3x_5 + 3x_6 + x_7 \le 11 \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# Solving this using R or Excel leads to the optimum $(1, 1, 0, 0, 1, 0, 0)$ with value 140.
#
# **Exercise 6.10** Verify that this is indeed the optimal solution by solving the problem in R and Excel.
#
# Although the solver seemed to have found the optimal answer without any problems, it required much more work. This becomes apparent when we solve big real-life ILO problems with hundreds or thousands of variables. To gain more insight in how ILOs are solved, let us have a look at Figure 6.9, where we see the steps to solve the knapsack example. We start with solving the LO relaxation, which is the problem without the integer or binary constraints (step 1). Sometimes we find an integer solution right away. Certain types of problems are even guaranteed to give integer solutions immediately. Here however $x_3$ is non-integer. Its value (150) is an upper bound to the best integer solution.

# %% [markdown]
# ![Figure 6.9: Solving an ILO problem](images/lecture8_fig6.9.png)

# %% [markdown]
# Now we branch on $x_3$, and we continue with the branch $x_3 = 0$. We solve the relaxation again, but with $x_3 = 0$. We find again a non-integer solution (step 2). We continue branching until we find an integer solution in step 4 with value 133. It is called a lower bound (LB) of the optimum: perhaps there are other integer solutions with values between 133 and 150. To find out if there are any such solutions we work our way back up to make sure all branches are dealt with. In step 5, we find an integer solution that is worse than the LB. In step 6, we find a higher binary value than the LB. It becomes the new LB, and the old LB is now sub-optimal (step 7). We have dealt with the left side of the tree, we move to the right. In step 8, we find a non-integer solution, we branch on $x_2$. In step 9 and 10, we find non-integer solutions which are worse or equal than the LB. Adding constraints will not make the value higher, therefore these branches can be discarded. We have dealt with all branches, and therefore the current LB is the optimum (step 11). This algorithm is called branch-and-bound.
#
# Many LO solvers can also handle integer constraints. However, not all solvers can solve big instances. The best solvers are proprietary, notably CPLEX and Gurobi.
#
# **Exercise 6.11** Solve by branch-and-bound the knapsack problem having rewards (15, 9, 10, 5), sizes (1, 3, 5, 4) and capacity 8. Check the result with R.
