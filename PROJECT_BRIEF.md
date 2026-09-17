# Deep-learning uncertainty quantification — project brief

This document is the project brief, formerly named `goal.md`. It records the agreed purpose, scientific scope, and working principles. Sections marked **proposed** are recommendations for discussion, not settled implementation requirements. They provide starting points for the literature review and pilots; they do not block the initial literature work. The new method's details remain deferred until integration. Git history records revisions, so the title does not carry a draft number.

# First agentic run

When asked to begin the project, start with a bounded first pass of stage 1: the literature review. This is the first discussion checkpoint within that stage, not a request to complete the entire review or roadmap in one run.

Produce the following project-local artifacts:

- An initial literature memo explaining the relevant method families, their treatment of supplied measurement errors, the function distributions they provide, and their applicability to the priority problems and related applications. Export it to readable HTML and PDF from a shared source, clearly labeled as an initial review.
- An evidence table with representative foundational and recent primary sources for each initial candidate family, including HMC-BNN, or an explicit record of unresolved coverage. Distinguish evidence for latent-function inference from evidence limited to noisy-observation prediction.
- A bibliography and search log recording sources, queries, dates, paper versions, access/reading status, and the search cutoff. Identify important missing or inaccessible evidence.
- A proposed baseline menu with reasons, an implementation order, and recommendations for the first experiment. Present choices and tradeoffs for discussion; the shortlist is settled collaboratively during stage 1.
- A short `PROGRESS.md` recording completed work, artifact locations, unresolved questions, and the recommended next bounded step.

Use minimal local tooling needed to organize sources and render the report. Benchmark implementation, model training, and cluster execution belong to later steps. The 24–48-hour experiment ceiling is a compute limit for experiments, not a requested duration for this literature pass.

The first run is complete when these artifacts form an inspectable initial comparison, substantive claims have source support or explicit evidence limitations, and a focused set of baseline/protocol decisions can be discussed with the user. Return for that discussion before finalizing the shortlist or starting stage 2. Subsequent literature passes deepen the review in the agreed directions.

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

Available resources: the current computer was inspected locally on 2026-09-17 using `/etc/os-release`, `uname -r`, `lscpu`, `free -h`, `df -h`, and `lspci -nn`. The additional desktop and cluster specifications were supplied by the user. Available memory and disk space are snapshots to recheck when scheduling work.

| Environment | Resources | Execution arrangement |
| --- | --- | --- |
| Current computer | Fedora Linux 44 KDE; kernel 7.2.5-200.fc44.x86_64; AMD Ryzen AI 7 445 with Radeon 840M, 6 cores / 12 threads; approximately 30 GiB RAM total and 24 GiB available at inspection; 8 GiB swap, unused; root/project filesystem approximately 952 GiB total with 914 GiB available; integrated AMD graphics (PCI identifies Krackan2), no discrete/NVIDIA GPU detected | The agent can execute locally |
| Additional desktop | Fedora Linux 44 KDE; kernel 7.2.5-200.fc44.x86_64; Ryzen 5 5600G, 6 cores / 12 threads, up to 4.47 GHz; 14 GiB RAM and 8 GiB swap; 237 GB root filesystem with approximately 139 GB available; integrated AMD Radeon Vega, no discrete/NVIDIA GPU | Available for agent-run work; establish access and the execution environment when using this machine |
| Cluster | H200 GPUs, Sapphire Rapids CPUs, and nodes with up to 2000 GB memory; a large pool of resources is available | No agent runs on the cluster. The agent prepares the work and the user schedules the jobs |

The agent should prepare a reproducible cluster job package: the code revision and environment instructions, input/configuration manifest, exact commands, requested CPU/GPU/memory/storage resources, runtime estimate and its basis, restart instructions, and expected output files. Once the scheduler is identified, include a suitable submission script. The user schedules execution and makes the results and logs available for the agent to validate and analyse. Scripts must run unattended, without requiring an agent on the compute node.

Each experiment specification should state what its budget covers, including methods, tuning, repetitions, and reference calculations. Node or device counts and aggregate CPU/GPU hours must be sized for the actual job; elapsed runtime alone does not define the allocation. Both local computers provide a CPU execution baseline, so initial validation and small pilots should not require a discrete GPU. Their memory and free disk space are planning constraints to recheck before substantial runs.

**Proposed budget convention:** the 24–48-hour ceiling covers the complete, predeclared experiment package, including its tuning, repetitions, and reference calculations. Splitting that package into jobs does not reset its allowance. Record queue waiting separately from execution time, and record aggregate resource use as well as elapsed runtime. Confirm the package scope and allocation before scheduling a large run.

**Agreed operating approach:** keep small validation and development runs inexpensive, use pilot timings to size larger comparisons, and save progress so that long experiments can resume after interruption. Report both elapsed runtime and allocated computational resources.

# Literature review

The review should be deep and focused on uncertainty quantification for scientific regression, linked observables, inverse problems, and related applications. The four stated problems are priorities, not an exhaustive boundary: similar problems should receive substantive attention where they offer relevant methods, comparisons, or insight. Also explore more distant applications selectively to identify transferable ideas or blind spots. Clearly distinguish methods covered in the literature review from those implemented in the benchmark; conclusions from these experiments apply to the tested settings.

Record search dates and a literature cutoff, search sources and queries, and reasons for including or excluding candidate methods. Include foundational work and relevant recent developments. A method's popularity, scientific relevance, assumptions, availability of an implementation, and computational feasibility should inform selection. Reach the final baseline shortlist collaboratively during stage 1, through discussion of the literature as it is explored. Method names elsewhere in this document are candidates, not a predetermined shortlist.

For each selected method, document its inference target, treatment of measurement noise, prior or regularization, supported outputs, computational requirements, and known limitations. Identify the exact variant and implementation used, with citations and any departures from the published algorithm. The specific algorithm behind labels such as "ensemble" or "repulsive ensemble" must be explicit.

Use survey papers to discover relevant work and primary sources to verify substantive methodological and empirical claims. Read the relevant full-text sections and supplements where available; search snippets and abstracts alone are insufficient to establish detailed claims or method rankings. Record whether a source was read in full, in relevant sections, or only at abstract level, and mark limited-access conclusions as provisional. Preserve DOI or arXiv identifiers, exact versions, publication/preprint status, and section, equation, figure, or table locations supporting the claims. Record code repository revisions when assessing implementations. References already present in this brief are methodological starting points, not a completed literature search.

Explicitly consider Hamiltonian Monte Carlo inference for Bayesian neural networks (HMC-BNN), including NUTS, among the methods to benchmark. It has two distinct potential roles: a method whose function uncertainty, computational cost, convergence, and limitations are evaluated, and a source of carefully checked numerical posterior references on sufficiently small problems. Its inclusion in the review and candidate comparisons must not be reduced to its reference role.

# Benchmark design

## First problem and reference hierarchy

Start with one-dimensional synthetic forward regression on a bounded interval, using the agreed noise model and treating input locations as exact. Evaluate on a fixed grid inside the observed input range. Specify the generating function, observation locations, network architecture, prior, likelihood, and evaluation quantities before running comparisons.

Use three distinct kinds of evidence:

1. **Exact posterior checks.** Begin with an analytically tractable model, such as Bayesian linear regression with fixed basis functions, to validate posterior calculations and metrics. This is a validation problem; its posterior must not be presented as the exact posterior of a different, fully trainable neural network.
2. **Numerical posterior references.** For a small neural network, construct an independently checked reference under the same model, prior, and likelihood as the approximation being evaluated. Hamiltonian Monte Carlo is a candidate. Check multiple chains, relevant function summaries, effective sample sizes, Monte Carlo error, and sampler diagnostics; an unresolved reference remains inconclusive. [Vehtari et al.](https://doi.org/10.1214/20-BA1221) provides methodological guidance on MCMC diagnostics.
3. **Known generating functions.** Use synthetic functions to measure reconstruction accuracy and uncertainty coverage, including controlled challenges to the assumed model. Knowing the generating function does not provide an exact posterior. Real-data examples can follow, with limits on what can be validated made explicit.

When HMC-BNN supplies a reference, distinguish that reference calculation and its budget from ordinary benchmark runs of the method. Do not report a reference run's agreement with itself as evidence of posterior accuracy. Validate the sampler on tractable cases, and use independent reference calculations and diagnostics where feasible, retaining the qualification that numerical references are approximate.

For inverse problems, specify boundary and initial conditions, fixed and inferred quantities, identifiability limitations, and solver tolerances. Check sensitivity to numerical resolution so that solver error can be distinguished from statistical uncertainty. Correct predictions of observables alone do not establish recovery of the latent function.

## Concrete starting experiment — proposed

The following gives a specific interpretation of the previously open choices of generating functions, architecture, prior, reference criteria, tuning, and repetitions. These are provisional defaults to refine during stage 1 and a small pilot, before fixing publication experiments.

| Choice | Proposed starting point |
| --- | --- |
| Inputs and evaluation | $x\in[-1,1]$, 25 equally spaced measurement locations including the endpoints, and a fixed 201-point evaluation grid in the same interval |
| Noise | Initially $\sigma_i=0.1$ for every observation. A first unequal-noise case uses $\sigma_i=0.05+0.1(x_i+1)/2$, supplied to each method |
| Exact validation problem | $h(x)=\beta_0+\beta_1x+\beta_2x^2$ with independent $\beta_j\sim\mathcal{N}(0,1)$ and the agreed Gaussian likelihood. For a fixed-function example, generate observations from $h_*(x)=0.5-0.7x+0.3x^2$. Compute the exact joint Gaussian posterior over function values |
| First neural-network problem | Generate observations from $h_*(x)=\sin(\pi x)+0.3x$. Use one scalar input, one hidden layer of 8 tanh units, and one linear output. This small model is for validating comparisons before scaling up |
| Neural-network prior | Independent zero-mean Gaussian weights with variance $1/n_{\mathrm{in}}$ for a layer with $n_{\mathrm{in}}$ inputs, and unit-variance Gaussian biases. Inspect prior function draws and document any revision before comparing inference methods; this is a candidate prior, not a claim of optimality |
| Numerical reference pilot | HMC with adaptive trajectory length (NUTS), initially 4 chains, each with 1000 warmup iterations and 1000 retained draws. Extend or revise the run according to diagnostics, accuracy, and budget; draw count alone does not qualify a reference |
| Tuning pilot | Use separate development datasets, at most 10 candidate configurations per method within a common tuning resource allowance, and validation observations for selection. Freeze the selected settings before final evaluation. Fix the target prior and likelihood for matched-posterior comparisons |
| Repetition pilot | Start with 5 independent evaluation datasets and 3 training/sampling seeds per approximate method. A reference fit uses multiple chains on each dataset; chains are not independent data replicates. Determine publication repetition counts from pilot variability and the desired precision of comparisons |

For accepting the numerical reference, propose rank-normalized $\widehat R<1.01$, bulk and tail effective sample sizes of at least 400 for the declared function quantities, and Monte Carlo standard errors of function means below 5% of their posterior standard deviations. Require no post-warmup divergences, inspect energy and trajectory diagnostics, and investigate disagreement between chains or parameter diagnostics, including neural-network symmetries. The multiple-chain and convergence checks follow [Stan's diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html); the precision target is a proposed project choice. These checks are necessary evidence, not proof of an exact posterior, and tighter tolerances may be needed for small method differences or tail claims.

For calibration averaged over the Bayesian model, generate functions from the specified prior and then generate noisy observations. Keep that experiment distinct from coverage under the two fixed generating functions above. If the reference cannot be validated within budget, reduce the validation problem or report the affected comparison as unresolved.

## Initial joint-function checks — proposed

Pointwise uncertainty describes $h(x)$ at individual locations. Joint uncertainty also describes how values at different locations vary together. Two methods can give similar pointwise intervals but different uncertainty for an integral or a difference between locations.

Start with a small set of joint checks: preserve coherent function draws over the evaluation grid, compare covariances on a fixed 21-point subgrid, and compare the distributions of $I=\int_{-1}^{1}h(x)\,dx$ and $\Delta=h(0.5)-h(-0.5)$ against the corresponding reference. Check numerical integration by refining the grid. This adds a concrete test of cross-location dependence while leaving simultaneous function bands and more elaborate function summaries for later stages.

## Comparison rules

Use the same data realizations, evaluation domains, and available measurement-error information across methods. Share architectures where meaningful and document exceptions. Posterior-fidelity comparisons require a shared target posterior; methods with different targets can still be compared for practical performance, with the distinction reported.

Set tuning budgets and stopping rules explicitly. Keep final evaluation results and synthetic ground truth separate from tuning; use designated development problems or validation data. Freeze and version the protocol for reported comparisons, while labeling exploratory runs separately. Include failed runs and failure rates rather than reporting only successful cases.

Repeat across independently generated datasets and training or sampling seeds. Keep these sources of randomness separate, and use paired comparisons on the same datasets where possible. Report variability across runs and Monte Carlo uncertainty in the metrics; correlated evaluation-grid points and samples from one fitted posterior must not be counted as independent dataset replicates. Repetition counts and numerical tolerances should be chosen after a pilot study, within the agreed compute budget.

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

Design the benchmark so that new methods, including mine, can be added without major refactoring.

Initial method interfaces should be based on the requirements of the benchmark and established methods. Any additional requirements specific to my method will be discussed when it is integrated.

Keep all project files, including agent instructions, source material records, code, and reports, within this repository.

# Roadmap — proposed

The progression from an exact validation problem to a small neural-network comparison and then broader problems is agreed. The detailed staging below remains a proposal.

There is no rush. Develop the project collaboratively in manageable stages, reviewing the results and revising the next stage before expanding its scope. These phases describe the intended progression, not authorization to begin implementing the entire project during document refinement.

## Gradual method development

Increase method complexity separately from problem complexity. The proposed implementation order is:

| Step | Candidate methods and scope | Condition for expanding |
| --- | --- | --- |
| A. Basic neural baselines | Establish ordinary network fitting as a sanity check, then independent ensembles and MC dropout if selected in stage 1 | A small forward-regression comparison with checked outputs, basic metrics, reproducibility, and reports |
| B. Additional approximate inference | Add simple, clearly specified variants of shortlisted methods, such as Laplace approximations or variational inference, one at a time | Each addition is checked on the same small problems before its comparisons are expanded |
| C. More demanding methods and variants | Develop HMC-BNN comparisons and selected interacting or repulsive ensembles, with their tuning, diagnostics, and computational costs | Small-case validation and measured cost justify larger or harder experiments |

This is a proposed development sequence; the literature review should refine the order according to the selected variants and implementation effort. A limited HMC-BNN reference run may be introduced after the basic neural baselines work, before completing steps B and C, to support posterior comparisons. Broader HMC-BNN testing and scaling studies follow gradually. The review can examine all candidate families from the outset while code development remains incremental.

For every new method, first demonstrate it on an existing simple benchmark, inspect its results and diagnostics, and then extend it to more difficult problems. Avoid introducing an unvalidated method and a new difficult problem in the same step. Expansion to every method–problem combination is not required; the literature, evidence, and compute budget should guide coverage.

## Project stages

| Stage | Work | Reviewable result |
| --- | --- | --- |
| 1. Focused literature review and protocol | Study methods relevant to the priority applications and similar problems in depth, scan more distant areas for blind spots, discuss candidates with the user, and settle the baseline shortlist, implementation order, and first experiment | A cited evidence table covering assumptions, measurement-noise treatment, capabilities, limitations, and implementations; an agreed shortlist and versioned protocol. Record the search cutoff at this stage |
| 2. Exact validation and reporting | Implement the tractable regression problem, analytic reference, core metrics, and function-sample storage on a local CPU; extend the literature-report tooling to computed results | Independently checked posterior means and covariances, checks of metrics and joint summaries, and reproducible HTML/PDF reports. Resolve discrepancies before neural-network comparisons |
| 3. Small neural-network pilot | Establish the simplest agreed neural baselines first, then a limited HMC-BNN numerical reference on the same small problem; measure runtime, memory, tuning sensitivity, and variability | A reference diagnostic report, paired baseline comparisons, and measured estimates for larger runs. Review or simplify unresolved cases; use the pilot to set final settings and repetition counts |
| 4. Forward-regression comparisons | Introduce the remaining shortlisted methods incrementally, following the method-development sequence; expand validated comparisons to problems 1 and 2, including unequal measurement errors and explicit input dependence | Repeated comparisons of posterior approximation, practical uncertainty, and cost, including HMC-BNN where selected and feasible. Prepare user-submitted cluster packages when local execution is insufficient; validate returned artifacts before updating reports |
| 5. Inverse-problem comparisons | Introduce problem 3 with a simple known relation, then a known dynamical model and problem 4; state identifiability, boundary/initial conditions, and numerical error checks. Use PINNs only where justified by the problem | Results for the primary unknown functions and associated observables, with validated forward calculations and explicit limitations. Apply the established comparison and provenance protocol |
| 6. New-method integration | When the user is ready, discuss the algorithm, add it through the method interface, revisit its closest related work, and run matched comparisons and justified ablations | Evidence for its contribution and limitations under the established protocol. This stage can begin after stage 3 once the benchmark is stable; it need not wait for all extensions in stages 4–5 |
| 7. Paper-supporting experiment release | Refresh the literature search, resolve or label outstanding issues, freeze selected code/configurations/results, and regenerate the reports and reusable figures/tables | An auditable experiment release with reproduction commands, sources, uncertainty on comparisons, and evidence supporting the claims to be used in the paper |

Candidate families for stage 1 include independent ensembles, MC dropout, variational inference, Laplace approximations, HMC/NUTS inference for Bayesian neural networks, and explicitly specified randomized or repulsive ensembles. HMC-BNN is a candidate comparison method as well as a possible numerical reference. The list is illustrative and can change during the review. Each stage ends with a report and discussion that determines the next bounded step; the plan does not require implementing every candidate or every possible stress test.

# Decisions still open

None of the following prevents the first literature pass. Resolve each decision when its supporting evidence is available and before the work that depends on it.

- Discuss or revise the concrete starting experiment and initial joint-function checks proposed above. These replace the earlier unspecified requests for functions, architecture, priors, diagnostics, tuning, repetitions, and joint uncertainty.
- Discuss or revise the detailed seven-stage roadmap, its gradual method-development sequence, and the proposed experiment-package budget convention.
- Resolve during stage 1: the final baseline shortlist through literature discussions, the recorded literature cutoff, exact metric definitions, and refinements to the initial experiment. The review's depth and application focus are agreed.
- Resolve after the pilot: publication repetition counts and numerical precision targets, based on observed variability and computational cost.
- Resolve when preparing cluster work: scheduler and submission conventions, specific resource allocations and package scope, software environment, and transfer of inputs/results. The user schedules cluster jobs; the agent does not run there.
