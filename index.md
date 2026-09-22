# Lecture Notes on Simulation and Optimization

These lecture notes are part of the broader course on Statistics, Simulation and Optimization (SSO) at the University of Amsterdam (UvA). They support the lectures, slides, tutorials, and assignments of the simulation and optimization component of the course. Large parts of these lecture notes are based on the book [*An Introduction to Business Analytics*](https://www.amazon.nl/-/en/Introduction-Business-Analytics-Ger-Koole/dp/9082017938) by Ger Koole written in 2019.

## About the simulation and optimization course part

The simulation and optimization part of this course is about turning big data into effective future decisions using mathematical modeling and optimization. Different quantitative methods are discussed to solve optimization problems in both deterministic and stochastic settings. Simulation techniques are used to evaluate models realistically in uncertain environments.

The course covers the following topics:

- (Integer) linear optimization/programming and commercial solvers
- Algorithms, heuristics, and complexity theory
- Monte Carlo simulation of functions of random variables
- Discrete-event simulation
- Statistical analysis of simulation results
- Simulation optimization techniques

## Learning objectives

On successful completion of this part of the course, participants are able to:

- Understand the importance of optimization and simulation in a big-data era
- Understand the role of statistics and data science in simulation and optimization
- Recognize, classify, and model optimization problems of deterministic and stochastic nature
- Solve optimization problems using algorithms and solvers in Python
- Translate optimization solutions into real-life decisions and critically reflect on them
- Simulate and statistically analyze random processes using Python and dedicated simulation software such as Arena and SimQuick
- Reflect on the importance of addressing uncertainty in light of the “flaw of averages”
- Recognize optimization and simulation opportunities in real life
- Apply optimization and simulation in real life in an ethically well-considered way
- Find and comprehend further simulation and optimization resources for problem-solving in practice

## Prerequisites

This course assumes a basic understanding of quantitative methods, including algebra, probability, statistics, and introductory data analysis. Familiarity with Python is beneficial, as many examples and exercises are implemented in Python.

## How to use these lecture notes

These notes are intended to support the lectures and other course materials rather than replace them. They are best read sequentially, with attention to the examples, exercises, and computational tasks. Students are encouraged to work through the examples actively and to connect the mathematical concepts to practical decision problems. Try the exercises first before looking at the solutions.

The materials are organized by lecture, and each lecture is divided into topic-specific notebooks. The table of contents in the left-hand menu follows this structure: lectures are grouped together, and the individual topics are listed underneath each lecture.

## Tooling

The simulation and optimization part of this course uses Python. Python's built-in functions are deliberately minimal; the data-analysis functionality comes from libraries, most importantly `numpy` (arrays and numerical computing), `pandas` (tabular data), `matplotlib` (plotting), and `scipy` (statistics and scientific computing). For optimization we add `pulp` (see [Modeling Tools and Solvers](notebooks/lecture10_modeling-tools.ipynb)). These notes are themselves Jupyter notebooks, combining code, its output, and explanation.

### Running the notebooks

Every notebook has an "Open in Colab" button at the top, which opens it directly in [Google Colab](https://colab.research.google.com/): a free, browser-based Jupyter environment that needs no local installation. Any packages a notebook needs beyond what Colab already provides (mainly `pulp`) are installed automatically the first time you run it there.

To run the notebooks on your own machine instead, clone the [GitHub repository](https://github.com/UvA-SSO/simopt-lecture-notes) and install the dependencies into an isolated Python environment. Any environment manager works (conda, venv, ...); these notes use [uv](https://docs.astral.sh/uv/). The course materials themselves are the project, while JupyterLab is only needed as a local editor for working with notebooks:

```sh
git clone https://github.com/UvA-SSO/simopt-lecture-notes.git
cd simopt-lecture-notes
uv sync --group dev  # installs the project dependencies and the local notebook editor
uv run jupyter lab  # Windows / macOS / Linux
```

These commands open Jupyter Lab so you can open a notebook under `notebooks/`.

## Acknowledgements

These lecture notes are based on Ger Koole's book *An Introduction to Business Analytics* (2019) and the lectures by Joost Berkhout. The notes were compiled with the help of Claude (Anthropic). Joost Berkhout takes full responsibility for the materials.
