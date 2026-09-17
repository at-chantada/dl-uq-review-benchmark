# Project brief — working third draft

This document records the agreed purpose and scientific scope. Sections marked **proposed** are recommendations for discussion, not settled implementation requirements. The new method's details remain deferred until integration.

# Context

I have developed a new algorithm for uncertainty quantification in deep learning. It falls under the category of ensemble methods.

The ultimate goal is to publish a paper presenting this method. This project will provide an important part of the evidence for that paper: a review of relevant current uncertainty quantification techniques and reproducible comparisons against established, widely known methods in the literature.

The method's intended contribution is to account explicitly for the supplied measurement uncertainties when constructing ensemble estimates of the unknown function. Its algorithmic details are deliberately deferred until the method is incorporated into the code. Assessment of its specific novelty and benefits will take place at that stage, informed by the literature review and benchmark.

The initial literature review and benchmark development should proceed around the agreed inference problems and evaluation objectives. They should not depend on disclosure of the new method or assumptions about its mechanism.

# Goal

Develop a review of relevant current uncertainty quantification techniques and a reproducible benchmark that can support comparisons in the eventual paper. The literature review should guide the selection of methods and experiments. The code should compare posterior approximation, practical uncertainty quality, convergence, computational cost, and sensitivity to tuning, and investigate limitations within the scientific scope below.

The deliverables are documented, extensible benchmark code and a clear, human-digestible HTML report and PDF explaining the reviewed methods, experimental design, results, and limitations. Provenance must make the evidence inspectable and reusable when preparing the paper.

# Evaluation objectives

The benchmark should evaluate two distinct objectives and report their results separately:

1. **Posterior approximation:** How closely does a method approximate the posterior over the unknown function under a specified Bayesian model? Each such comparison must identify the target model, prior, likelihood, and quantity being compared. Reference posteriors must be distinguished as analytically exact or numerically approximated, with validation evidence for numerical references. Methods that do not target the same posterior should have that distinction made explicit when interpreting discrepancies.
2. **Practical uncertainty quality:** How reliable and useful are a method's uncertainty estimates for the unknown function within the range of the data? This includes accuracy of the inferred function, calibration and informativeness of uncertainty, robustness within this domain, and computational cost. Good practical performance and close posterior approximation are separate findings; neither should be used as a substitute for evidence about the other.

The reference strategy and comparison principles below are agreed. Specific problem instances, metric definitions, and numerical settings remain to be selected during planning.

# Scientific scope

The primary target is the posterior distribution of the underlying unknown function: $y(x)$ in problem 1, $A(x)$ in problem 2, and $u(x)$ in problems 3 and 4. The related observable functions are also unknown and have uncertainty, as listed below; their governing relations are known. Inference must respect these relations, with evaluation focused on the primary target. Measurement noise enters the observation model; the targets are the underlying functions, not future noisy measurements. Evaluation is restricted to the range of the observed input data. Extrapolation is outside the scope of this project. Each benchmark must specify its evaluation domain; for multiple datasets, their respective input ranges and the chosen evaluation domain must be recorded explicitly.

The main applications are the following regression problems. Neural networks of specified architecture represent unknown functions as required by the chosen formulation. Other applications can be discussed if they strengthen the comparison.

| Problem | Observations and known relations | Unknown functions | Primary target |
| --- | --- | --- | --- |
| 1. Simple forward regression | Measurements of $y(x)$ | $y(x)$ | $y(x)$ |
| 2. Multiple forward regression | Measurements of $A(x)$ and $B(x)$, with $B(x)=f[A,x]$ | $A(x)$ and $B(x)$ | $A(x)$ |
| 3. Simple inverse regression | Measurements of $y(x)$, with $y(x)=f[u,x]$ | $u(x)$ and $y(x)$ | $u(x)$ |
| 4. Multiple inverse regression | Measurements of $A(x)$ and $B(x)$, with $A(x)=g[u,x]$ and $B(x)=f[A,x]$ | $u(x)$, $A(x)$, and $B(x)$ | $u(x)$ |

For a measured quantity $z$, a dataset is $\mathcal{D}_z=\{(x_{z,i},z_i,\sigma_{z,i})\}_{i=1}^{N_z}$. Different observables may have different input locations and numbers of measurements.

The notation $f[h,x]$ explicitly allows dependence on both the unknown function $h$ and the input $x$. It includes pointwise relations such as $B(x)=f(A(x),x)=x[A(x)]^2$, as well as relations involving derivatives, integrals, or solutions of known differential equations. The same convention applies to $g[u,x]$ and $f[u,x]$. Knowing a governing relation does not make the functions it relates known. When a secondary function is determined by that relation and the inferred quantities, its uncertainty is obtained by propagating their joint uncertainty through the relation; it need not be parameterized independently.

**Agreed initial noise model:** independent Gaussian measurement errors with known standard deviations, which may differ between observations:

$$
z_i=z(x_{z,i})+\epsilon_{z,i},\qquad \epsilon_{z,i}\sim\mathcal{N}(0,\sigma_{z,i}^2).
$$

Correlated errors, non-Gaussian errors, and uncertain noise levels are possible later extensions to discuss, not requirements for the first benchmark.

The governing equations defining the relations $f$ and $g$ are known. Evaluating those relations may require analytical or numerical solution of differential equations; learning unknown governing equations is outside the scope of this project. For example, $u(x)$ could be an unknown forcing function and $A(x)=g[u,x]$ the unknown position function of an oscillator governed by known equations. The primary task would be to infer the forcing from noisy position measurements, while also obtaining the associated uncertainty in the position function. Numerical approaches, potentially including PINNs (physics-informed neural networks), may be considered when solving or enforcing the known equations.

# Compute budget

The current extreme upper limit is approximately one to two days (24–48 hours) of cluster runtime per experiment. This is a ceiling, not a target for routine development runs. A later extension to approximately one week may be considered, but is outside the current budget.

The available cluster hardware and allocation are still unspecified. Each experiment specification should state what its budget covers, including methods, tuning, repetitions, and reference calculations. Node or device counts, memory, storage, and aggregate CPU/GPU hours remain to be determined; elapsed runtime alone does not define the total computational allocation.

**Agreed operating approach:** keep small validation and development runs inexpensive, use pilot timings to size larger comparisons, and save progress so that long experiments can resume after interruption. Report both elapsed runtime and allocated computational resources.

# Literature review

Start with a map of the main approaches to uncertainty quantification in deep learning, then examine methods relevant to the stated regression and inverse problems in greater depth. Clearly distinguish methods covered in the literature review from those implemented in the benchmark; conclusions from these experiments apply to the tested settings.

Record search dates and a literature cutoff, search sources and queries, and reasons for including or excluding candidate methods. Include foundational work and relevant recent developments. A method's popularity, scientific relevance, assumptions, availability of an implementation, and computational feasibility should inform selection. The initial map should lead to a justified shortlist rather than a requirement to implement every family.

For each selected method, document its inference target, treatment of measurement noise, prior or regularization, supported outputs, computational requirements, and known limitations. Identify the exact variant and implementation used, with citations and any departures from the published algorithm. The specific algorithm behind labels such as "ensemble" or "repulsive ensemble" must be explicit.

# Benchmark design

## First problem and reference hierarchy

Start with one-dimensional synthetic forward regression on a bounded interval, using the agreed noise model and treating input locations as exact. Evaluate on a fixed grid inside the observed input range. Specify the generating function, observation locations, network architecture, prior, likelihood, and evaluation quantities before running comparisons.

Use three distinct kinds of evidence:

1. **Exact posterior checks.** Begin with an analytically tractable model, such as Bayesian linear regression with fixed basis functions, to validate posterior calculations and metrics. This is a validation problem; its posterior must not be presented as the exact posterior of a different, fully trainable neural network.
2. **Numerical posterior references.** For a small neural network, construct an independently checked reference under the same model, prior, and likelihood as the approximation being evaluated. Hamiltonian Monte Carlo is a candidate. Check multiple chains, relevant function summaries, effective sample sizes, Monte Carlo error, and sampler diagnostics; an unresolved reference remains inconclusive. [Vehtari et al.](https://doi.org/10.1214/20-BA1221) provides methodological guidance on MCMC diagnostics.
3. **Known generating functions.** Use synthetic functions to measure reconstruction accuracy and uncertainty coverage, including controlled challenges to the assumed model. Knowing the generating function does not provide an exact posterior. Real-data examples can follow, with limits on what can be validated made explicit.

For inverse problems, specify boundary and initial conditions, fixed and inferred quantities, identifiability limitations, and solver tolerances. Check sensitivity to numerical resolution so that solver error can be distinguished from statistical uncertainty. Correct predictions of observables alone do not establish recovery of the latent function.

## Comparison rules

Use the same data realizations, evaluation domains, and available measurement-error information across methods. Share architectures where meaningful and document exceptions. Posterior-fidelity comparisons require a shared target posterior; methods with different targets can still be compared for practical performance, with the distinction reported.

Set tuning budgets and stopping rules explicitly. Keep final evaluation results and synthetic ground truth separate from tuning; use designated development problems or validation data. Freeze and version the protocol for reported comparisons, while labeling exploratory runs separately. Include failed runs and failure rates rather than reporting only successful cases.

Repeat across independently generated datasets and training or sampling seeds. Keep these sources of randomness separate, and use paired comparisons on the same datasets where possible. Report variability across runs and Monte Carlo uncertainty in the metrics. Repetition counts and numerical tolerances should be chosen after a pilot study, within the agreed compute budget.

Start stress tests by varying data count, measurement-noise scale, unequal error bars, and observation density within the data range. Later stages can investigate harder function shapes and weakly identifiable inverse problems. Extrapolation remains excluded.

## Initial metrics and interpretation

| Question | Candidate measurements |
| --- | --- |
| Accuracy of the inferred function | Error of the posterior or ensemble mean on the evaluation grid, with a stated aggregation rule |
| Agreement with a reference posterior | Errors in means, variances, and quantiles; distributional discrepancies; selected joint function summaries |
| Useful function uncertainty | Coverage and interval width at several nominal levels; continuous ranked probability score (CRPS) against known synthetic function values |
| Computational cost | Training and tuning time, prediction or sampling time, peak memory, and total work for all members or chains |
| Convergence and sensitivity | Changes with optimization effort, ensemble size, posterior sample count, and tuning choices, reported separately |

CRPS and interval scores are candidate tools for evaluating distributional and interval forecasts; see [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437). Report the target being scored explicitly. Scores for future noisy observations can be supporting diagnostics but do not replace evaluation of the underlying function. Density-based scores require a justified density representation; an empirical ensemble should not silently be treated as Gaussian.

Distinguish pointwise intervals from simultaneous bands over a function. Preserve coherent joint function draws where supported, so dependencies across input locations and uncertainty in integrals or other function summaries can be evaluated. A method that provides only marginal intervals should have that limitation recorded.

Separate repeated-data coverage for fixed generating functions from Bayesian calibration averaged over functions drawn from the model's prior. Candidate checks include [simulation-based calibration](https://arxiv.org/abs/1804.06788); test quantities need care because some checks can miss incorrect posteriors, as discussed by [Modrák et al.](https://doi.org/10.1214/23-BA1404). Neither nominal coverage in selected examples nor a successful calibration check alone establishes posterior accuracy.

# Provenance and reporting

For every reported experiment, retain a stable run identifier, complete configuration, code revision and any uncommitted changes, dependency versions, data or generator specification, seeds, hardware and numerical precision, solver settings where relevant, runtime, logs, and validation status. Save the function samples or other outputs needed to regenerate reported metrics and figures. Record data and artifact checksums to identify the exact inputs and results used.

Every figure and table should link to its source runs, processing code, and relevant checks. Every substantive literature claim should link to a source and, where useful, a section or equation. Distinguish published claims, our reproduced results, and our interpretations. This provenance should include failures, exclusions, and unresolved checks.

Generate HTML and PDF from a shared report source and saved results. Both formats should contain the same core argument, figures, and citations, with readable mathematics and static equivalents for any essential interactive content. Provide an accessible main narrative and detailed methodological material for inspection.

Document two reproduction paths: rebuilding the reports from saved results, and rerunning the underlying experiments. State expected resources and numerical tolerances rather than promising identical results on every machine. A milestone is ready for review when its figures are traceable, its checks are recorded, and both report formats can be rebuilt using documented commands.

# Implementation requirements

Use Python primarily, unless there is a clear reason to use another language. Preferred tools are uv, ruff, and pyrefly. Document the code and mathematical assumptions so that I can understand and review the implementation. Document the provenance of calculations, figures, results, and their checks.

Given that later on I want to incorporate my method into this benchmark, the code should be made with the capability of adding new methods without need of big refactors.

Initial method interfaces should be based on the requirements of the benchmark and established methods. Any additional requirements specific to my method will be discussed when it is integrated.

All relevant files are and should be located in this directory. Even things like AGENTS.md.

# Roadmap — proposed

The progression from an exact validation problem to a small neural-network comparison and then broader problems is agreed. The detailed staging below remains a proposal.

There is no rush. Develop the project collaboratively in manageable stages, reviewing the results and revising the next stage before expanding its scope. These phases describe the intended progression, not authorization to begin implementing the entire project during document refinement.

| Stage | Work | Reviewable result |
| --- | --- | --- |
| 1. Literature map and protocol | Review method families and select the first problem, baselines, metrics, and compute budget | Cited method map and an explicit first-experiment specification |
| 2. Exact validation problem | Implement a tractable posterior, core metrics, and provenance; establish report generation | A small validated example with traceable HTML and PDF reports |
| 3. First neural-network comparison | Add a small network, a checked numerical posterior reference, and a few baselines such as independent ensembles and MC dropout | A complete comparison under the agreed Gaussian noise model, with limitations and reference diagnostics |
| 4. Broader comparisons | Add justified methods and stress tests, then multiple-observable and inverse problems | Updated reports with repeated experiments, cost comparisons, and documented failure cases |
| 5. New-method integration | Discuss the new algorithm when I am ready, add it through the method interface, and revisit the closest related work | Comparisons under the established protocol and evidence for the paper's claims |

Potential additions after the initial baselines include variational inference, Laplace approximations, and explicitly specified randomized or repulsive ensembles. Their priority should come from the literature review and pilot results. Integration of the new method need not wait for every possible benchmark extension.

# Decisions still open

- Cluster hardware and allocation, memory and storage budgets, and the precise experiment scope covered by the 24–48-hour runtime ceiling.
- The first generating functions, network architecture and prior, numerical reference criteria, tuning protocol, and repetition counts.
- How much joint function uncertainty to assess in the first comparison, beyond pointwise summaries.
- The literature cutoff, depth of coverage outside the selected applications, and final baseline shortlist.
- Detailed staging of the roadmap, building on the agreed reference strategy and benchmark, provenance, and reporting principles.
