# Context

I have developed a new algorithm for uncertainty quantification in deep learning. It falls under the category of ensemble methods.

# Goal

To be able to know how my new method compares against what other people have done, and are using, I need to first have a fully working code that compares different methods. The goal would be to develop this code, which should: i) robustly compare the methods through several aspects, like convergence, accuracy, efficiency, speed, complexity, fine tuning, and such, ii) if possible, use examples where the ground truth predictive posterior is known to use as point of comparison between the methods, iii) test the limits of applicability of each method.


The code that achieves this is the main goal. However, it is important that the results are presented in a clear human-digestable manner. These should be an HTML page, and a PDF.

# Type of problem

Uncertainty quantification methods have very wide applicability, so I'll list the main types of problems I am interested in tackling here. However, I am interested in exploring other kinds of problems so that a more robust comparison can be achieved. From simplest to hardest, here are the types of problems that I am interested in:

  1. Simple forward regression: Given some dataset $\mathcal{D}:\{(x_i,y_i,\sigma_i) for i=1,...,N\}$, where $y_i$ is a measurement of $y(x)$ at $x=x_i$ with error $\sigma_i$, one wants to use an uncertainty quantification method with $\mathcal{D}$ to obtain the predictive posterior of $y(x)$ given the data, using a neural network of some given size and architecture as the function template.
  2. Multiple forward regression: Given two datasets $\mathcal{D}_A:\{(x_{A_i},A_i,\sigma_{A_i}) for i=1,...,N_A\}$ and $\mathcal{D}_B:\{(x_{B_i},B_i,\sigma_{B_i}) for i=1,...,N_B\}$, with $B=f[A(x),x]$, one wants to use an uncertainty quantification method with $\mathcal{D}_A$ and $\mathcal{D}_B$ to obtain the predictive posterior of $A(x)$ given the data, using a neural network of some given size and architecture as the function template.
  3. Simple inverse regression: Given some dataset $\mathcal{D}:\{(x_i,y_i,\sigma_i) for i=1,...,N\}$, where $y_i$ is a measurement of $y(x)$ at $x=x_i$ with error $\sigma_i$, one wants to use an uncertainty quantification method with $\mathcal{D}$ to obtain the predictive posterior of $u(x)$, which relates to the observable as $y=f[u(x),x]$ given the data, using a neural network of some given size and architecture as the function template.
  4. Multiple forward regression: Given two datasets $\mathcal{D}_A:\{(x_{A_i},A_i,\sigma_{A_i}) for i=1,...,N_A\}$ and $\mathcal{D}_B:\{(x_{B_i},B_i,\sigma_{B_i}) for i=1,...,N_B\}$, with $B=f[A(x),x]$, and $A=g[u(x),x]$ one wants to use an uncertainty quantification method with $\mathcal{D}_A$ and $\mathcal{D}_B$ to obtain the predictive posterior of $u(x)$ given the data, using a neural network of some given size and architecture as the function template.

The functions $f$ and $g$ could either be known, or need to be solved for. An example would be that $u(x)$ is some unknown forcing function, and $A=f[u(x),x]$ is some observable quantity that is affected by that forcing function. Then, the appropiate model that relates the two would be a differential equation. $A_i$ could be mesurements of the position of an oscillator at different times, and one wants to know the forcing that the oscillator is experiencing. Of course, this example has an analytical solution, but one could think cases with this is not the case, and PINNs (physics-informed neural networks) could be employed for the uncertainty quantification to work.

# Guidelines

The code should mainly be in python, unless there is a good reason to use other languages. Regarding python tools, I like uv, ruff, and pyrefly. The code should be well documented. I would like to understand the final product as much as possible, and be able to review it somewhat. Document the provenance of every calculation, figure, result, and how these were check.

Given that later on I want to incorporate my method into this benchmark, the code should be made with the capability of adding new methods without need of big refactors.

All relevant files are and should be located in this directory. Even things like AGENTS.md.

# Plan

There is no big hurry to make this. I would like to make this alongside the agents by steps. The rough steps should be planned ahead. That is what this document is for, to starting sketching out the roadmap. A rough idea I have for the roadmap is to start with simple methods like regular ensemble, and MC dropout, and think of which metrics would be the simplest to do first. Then, slowly make things more sophisticated. Add methods like Hamilton Monte Carlo, and repulsive ensembles. And consider more complicated and robust test to compare the methods.
