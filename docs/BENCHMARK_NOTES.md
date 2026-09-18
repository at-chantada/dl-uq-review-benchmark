# Benchmark design and development notes

Read when planning or assessing benchmark experiments. [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) governs the project and the scope of each run; this document supplies supporting detail, not an implementation assignment.

The validation and comparison safeguards apply whenever relevant work is undertaken. The progression from exact validation to a small neural network is a current working agreement. Sections marked **proposed**, candidate metrics, method variants, numerical defaults, and detailed stage contents can be revised in light of the literature and pilots. Record revisions and their reasons; discuss material changes to the agreed scope before acting on them. Method details specific to the user's new ensemble remain deferred until integration.

The methodological references below are starting points, not a completed literature review.

## First problem and reference hierarchy

Start with one-dimensional synthetic forward regression on a bounded interval, using the agreed noise model and treating input locations as exact. Evaluate on a fixed grid inside the observed input range. Specify the generating function, observation locations, network architecture, prior, likelihood, and evaluation quantities before running comparisons.

Use three distinct kinds of evidence:

1. **Exact posterior checks.** Begin with an analytically tractable model, such as Bayesian linear regression with fixed basis functions, to validate posterior calculations and metrics. This is a validation problem; its posterior must not be presented as the exact posterior of a different, fully trainable neural network.
2. **Numerical posterior references.** For a small neural network, construct an independently checked reference under the same model, prior, and likelihood as the approximation being evaluated. Hamiltonian Monte Carlo is a candidate. Check multiple chains, relevant function summaries, effective sample sizes, Monte Carlo error, and sampler diagnostics; an unresolved reference remains inconclusive. [Vehtari et al.](https://doi.org/10.1214/20-BA1221) provides methodological guidance on MCMC diagnostics.
3. **Known generating functions.** Use synthetic functions to measure reconstruction accuracy and uncertainty coverage, including controlled challenges to the assumed model. Knowing the generating function does not provide an exact posterior. Real-data examples can follow, with limits on what can be validated made explicit.

When HMC-BNN supplies a reference, distinguish that reference calculation and its budget from ordinary benchmark runs of the method. Do not report a reference run's agreement with itself as evidence of posterior accuracy. Validate the sampler on tractable cases, and use independent reference calculations and diagnostics where feasible, retaining the qualification that numerical references are approximate.

For inverse problems, specify boundary and initial conditions, fixed and inferred quantities, identifiability limitations, and solver tolerances. Check sensitivity to numerical resolution so that solver error can be distinguished from statistical uncertainty. Correct predictions of observables alone do not establish recovery of the latent function.

For example, $u(x)$ could be an unknown forcing function and $A(x)=g[u,x]$ the unknown position function of an oscillator governed by known equations. Infer the forcing from noisy position measurements while also obtaining the associated position uncertainty. Numerical approaches, including PINNs where justified, may solve or enforce the known equations.

## Concrete starting experiment — proposed

These provisional defaults cover generating functions, architecture, prior, reference criteria, tuning, and repetitions. Refine them during the literature review and small pilots before fixing publication experiments.

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

## Gradual method development — proposed

Increase method complexity separately from problem complexity. The proposed implementation order is:

| Step | Candidate methods and scope | Condition for expanding |
| --- | --- | --- |
| A. Basic neural baselines | Establish ordinary network fitting as a sanity check, then independent ensembles and MC dropout if selected in stage 1 | A small forward-regression comparison with checked outputs, basic metrics, reproducibility, and reports |
| B. Additional approximate inference | Add simple, clearly specified variants of shortlisted methods, such as Laplace approximations or variational inference, one at a time | Each addition is checked on the same small problems before its comparisons are expanded |
| C. More demanding methods and variants | Develop HMC-BNN comparisons and selected interacting or repulsive ensembles, with their tuning, diagnostics, and computational costs | Small-case validation and measured cost justify larger or harder experiments |

This is a proposed development sequence; the literature review should refine the order according to the selected variants and implementation effort. A limited HMC-BNN reference run may be introduced after the basic neural baselines work, before completing steps B and C, to support posterior comparisons. Broader HMC-BNN testing and scaling studies follow gradually. The review can examine all candidate families from the outset while code development remains incremental.

For every new method, first demonstrate it on an existing simple benchmark, inspect its results and diagnostics, and then extend it to more difficult problems. Avoid introducing an unvalidated method and a new difficult problem in the same step. Expansion to every method–problem combination is not required; the literature, evidence, and compute budget should guide coverage.

## Project stages — proposed

| Stage | Work | Reviewable result |
| --- | --- | --- |
| 1. Focused literature review and protocol | Study methods relevant to the priority applications and similar problems in depth, scan more distant areas for blind spots, discuss candidates with the user, and settle the baseline shortlist, implementation order, and first experiment | A cited evidence table covering assumptions, measurement-noise treatment, capabilities, limitations, and implementations; an agreed shortlist and versioned protocol. Record the search cutoff at this stage |
| 2. Exact validation and reporting | Implement the tractable regression problem, analytic reference, core metrics, and function-sample storage on a local CPU; extend the literature-report tooling to computed results | Independently checked posterior means and covariances, checks of metrics and joint summaries, and reproducible HTML/PDF reports. Resolve discrepancies before neural-network comparisons |
| 3. Small neural-network pilot | Establish the simplest agreed neural baselines first, then a limited HMC-BNN numerical reference on the same small problem; measure runtime, memory, tuning sensitivity, and variability | A reference diagnostic report, paired baseline comparisons, and measured estimates for larger runs. Review or simplify unresolved cases; use the pilot to set final settings and repetition counts |
| 4. Forward-regression comparisons | Introduce the remaining shortlisted methods incrementally, following the method-development sequence; expand validated comparisons to problems 1 and 2, including unequal measurement errors and explicit input dependence | Repeated comparisons of posterior approximation, practical uncertainty, and cost, including HMC-BNN where selected and feasible. Prepare user-submitted cluster packages when local execution is insufficient; validate returned artifacts before updating reports |
| 5. Inverse-problem comparisons | Introduce problem 3 with a simple known relation, then a known dynamical model and problem 4; state identifiability, boundary/initial conditions, and numerical error checks. Use PINNs only where justified by the problem | Results for the primary unknown functions and associated observables, with validated forward calculations and explicit limitations. Apply the established comparison and provenance protocol |
| 6. New-method integration | When the user is ready, discuss the algorithm, add it through the method interface, revisit its closest related work, and run matched comparisons and justified ablations | Evidence for its contribution and limitations under the established protocol |
| 7. Paper-supporting experiment release | Refresh the literature search, resolve or label outstanding issues, freeze selected code/configurations/results, and regenerate the reports and reusable figures/tables | An auditable experiment release with reproduction commands, sources, uncertainty on comparisons, and evidence supporting the claims to be used in the paper |

Candidate families for stage 1 include independent ensembles, MC dropout, variational inference, Laplace approximations, HMC/NUTS inference for Bayesian neural networks, and explicitly specified randomized or repulsive ensembles. HMC-BNN is a candidate comparison method as well as a possible numerical reference. The list is illustrative and can change during the review. Stages contain multiple bounded assignments, each followed by a review checkpoint under the brief's collaboration rules. A stage description does not authorize completing all its work in one run. The plan does not require implementing every candidate or every possible stress test.

## Decisions to resolve when needed

- During literature discussions: the baseline shortlist and implementation order, metric definitions, and the initial experiment, including its prior and joint-function checks.
- After small pilots: tuning allowances, publication repetition counts, and numerical precision targets, based on variability and measured cost.
- Before reported comparisons: a versioned protocol with targets, settings, budgets, and acceptance criteria fixed separately from exploratory runs.

These decisions do not block the initial literature orientation.
