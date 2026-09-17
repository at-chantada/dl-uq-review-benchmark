# Context

I have developed a new algorithm for uncertainty quantification in deep learning. It falls under the category of ensemble methods.

The ultimate goal is to publish a paper presenting this method. This project will provide an important part of the evidence for that paper: a review of relevant current uncertainty quantification techniques and reproducible comparisons against established, widely known methods in the literature.

The method's intended contribution is to account explicitly for the supplied measurement uncertainties when constructing ensemble estimates of the unknown function. Its algorithmic details are deliberately deferred until the method is incorporated into the code. Assessment of its specific novelty and benefits will take place at that stage, informed by the literature review and benchmark.

The initial literature review and benchmark development should proceed around the agreed inference problems and evaluation objectives. They should not depend on disclosure of the new method or assumptions about its mechanism.

# Goal

To be able to know how my new method compares against what other people have done, and are using, I need to first have a fully working code that compares different methods. The goal would be to develop this code, which should: i) robustly compare the methods through several aspects, like convergence, accuracy, efficiency, speed, complexity, fine tuning, and such, ii) if possible, use examples where the exact posterior over the unknown function is known to use as a point of comparison between the methods, iii) test the limits of applicability of each method within the problem scope defined below.


The literature review should guide the selection of comparison methods and experiments. The benchmark code is a central research deliverable supporting the eventual paper, and should help establish the new method's strengths, limitations, and scope of applicability.

The reviewed methods, comparison design, and results should be presented in a clear, human-digestible HTML report and PDF, with provenance that makes the evidence inspectable and reusable when preparing the paper.

# Evaluation objectives

The benchmark should evaluate two distinct objectives and report their results separately:

1. **Posterior approximation:** How closely does a method approximate the posterior over the unknown function under a specified Bayesian model? Each such comparison must identify the target model, prior, likelihood, and quantity being compared. Reference posteriors must be distinguished as analytically exact or numerically approximated, with validation evidence for numerical references. Methods that do not target the same posterior should have that distinction made explicit when interpreting discrepancies.
2. **Practical uncertainty quality:** How reliable and useful are a method's uncertainty estimates for the unknown function within the range of the data? This includes accuracy of the inferred function, calibration and informativeness of uncertainty, robustness within this domain, and computational cost. Good practical performance and close posterior approximation are separate findings; neither should be used as a substitute for evidence about the other.

The specific metrics, reference problems, and evaluation protocols for both objectives remain to be agreed during planning.

# Type of problem

The target is the posterior distribution of the underlying unknown function: $y(x)$ in problem 1, $A(x)$ in problem 2, and $u(x)$ in problems 3 and 4. Measurement noise enters the observation model; the target is the underlying function, not a future noisy measurement. Evaluation is restricted to the range of the observed input data. Extrapolation is outside the scope of this project. Each benchmark must specify its evaluation domain; for multiple datasets, their respective input ranges and the chosen evaluation domain must be recorded explicitly.

Uncertainty quantification methods have very wide applicability, so I'll list the main types of problems I am interested in tackling here. However, I am interested in exploring other kinds of problems so that a more robust comparison can be achieved. From simplest to hardest, here are the types of problems that I am interested in:

  1. Simple forward regression: Given some dataset $\mathcal{D}:\{(x_i,y_i,\sigma_i) for i=1,...,N\}$, where $y_i$ is a measurement of $y(x)$ at $x=x_i$ with error $\sigma_i$, one wants to use an uncertainty quantification method with $\mathcal{D}$ to obtain the posterior of $y(x)$ given the data, using a neural network of some given size and architecture as the function template.
  2. Multiple forward regression: Given two datasets $\mathcal{D}_A:\{(x_{A_i},A_i,\sigma_{A_i}) for i=1,...,N_A\}$ and $\mathcal{D}_B:\{(x_{B_i},B_i,\sigma_{B_i}) for i=1,...,N_B\}$, with $B=f[A(x),x]$, one wants to use an uncertainty quantification method with $\mathcal{D}_A$ and $\mathcal{D}_B$ to obtain the posterior of $A(x)$ given the data, using a neural network of some given size and architecture as the function template.
  3. Simple inverse regression: Given some dataset $\mathcal{D}:\{(x_i,y_i,\sigma_i) for i=1,...,N\}$, where $y_i$ is a measurement of $y(x)$ at $x=x_i$ with error $\sigma_i$, one wants to use an uncertainty quantification method with $\mathcal{D}$ to obtain the posterior of $u(x)$, which relates to the observable as $y=f[u(x),x]$ given the data, using a neural network of some given size and architecture as the function template.
  4. Multiple inverse regression: Given two datasets $\mathcal{D}_A:\{(x_{A_i},A_i,\sigma_{A_i}) for i=1,...,N_A\}$ and $\mathcal{D}_B:\{(x_{B_i},B_i,\sigma_{B_i}) for i=1,...,N_B\}$, with $B=f[A(x),x]$, and $A=g[u(x),x]$ one wants to use an uncertainty quantification method with $\mathcal{D}_A$ and $\mathcal{D}_B$ to obtain the posterior of $u(x)$ given the data, using a neural network of some given size and architecture as the function template.

The governing equations defining the relations $f$ and $g$ are known. Evaluating those relations may require analytical or numerical solution of differential equations; learning unknown governing equations is outside the scope of this project. For example, $u(x)$ could be an unknown forcing function and $A=g[u(x),x]$ the position of an oscillator governed by known equations. The task would be to infer the forcing from noisy position measurements. Numerical approaches, potentially including PINNs (physics-informed neural networks), may be considered when solving or enforcing the known equations.

# Guidelines

The code should mainly be in python, unless there is a good reason to use other languages. Regarding python tools, I like uv, ruff, and pyrefly. The code should be well documented. I would like to understand the final product as much as possible, and be able to review it somewhat. Document the provenance of every calculation, figure, result, and how these were check.

Given that later on I want to incorporate my method into this benchmark, the code should be made with the capability of adding new methods without need of big refactors.

Initial method interfaces should be based on the requirements of the benchmark and established methods. Any additional requirements specific to my method will be discussed when it is integrated.

All relevant files are and should be located in this directory. Even things like AGENTS.md.

# Plan

There is no big hurry to make this. I would like to make this alongside the agents by steps. The rough steps should be planned ahead. That is what this document is for, to starting sketching out the roadmap. A rough idea I have for the roadmap is to start with simple methods like regular ensemble, and MC dropout, and think of which metrics would be the simplest to do first. Then, slowly make things more sophisticated. Add methods like Hamilton Monte Carlo, and repulsive ensembles. And consider more complicated and robust test to compare the methods.
