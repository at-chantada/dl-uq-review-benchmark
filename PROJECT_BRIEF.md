# Deep-learning uncertainty quantification — project brief

Repository: [at-chantada/dl-uq-review-benchmark](https://github.com/at-chantada/dl-uq-review-benchmark)

This is the canonical brief for the project. Read it before project work; consult the linked supporting documents when their details become relevant. The roadmap describes possible future work. The user's current assignment determines what to do now.

## Purpose and core commitments

The ultimate goal is a paper presenting the user's new ensemble method. This project supplies a deep review of relevant current uncertainty quantification techniques in deep learning and a reproducible, extensible benchmark against established methods.

Preserve these commitments throughout development:

- **Scientific purpose:** build credible evidence for the paper, including limitations and negative results. Assess the new method's novelty and benefits from evidence.
- **Scientific validity:** define inference targets, make fair comparisons, validate calculations, and distinguish what the evidence establishes from what remains uncertain.
- **Provenance:** make literature claims and computed results traceable to their sources, configurations, code, and checks, including failures and exclusions.
- **Human-digestible reporting:** provide readable HTML and PDF from a shared source, with understandable mathematics, figures, citations, and explanations.

### What can change

| Status | How to use it |
| --- | --- |
| Core commitments | Preserve the purpose and evidence standards above. Changes to the project's purpose or promised deliverables require explicit user direction. |
| Current working agreements | The scientific scope, evaluation objectives, initial noise model, compute ceiling, and gradual development approach below guide current work. They can be revisited with reasons and discussion; they remain active until revised. |
| Provisional design choices | Candidate methods, numerical settings, metrics, software structure, and detailed sequencing are starting points to improve through the literature and pilots. They are not a fixed specification or a requirement to implement every option. |

Evidence-based improvements are expected. Record substantive changes and their rationale in the relevant document and progress notes. Discuss changes that alter the inference target, agreed scope, resource allowance, or next deliverable before acting on them. Routine choices within the agreed assignment can be made autonomously.

The new method is intended to account explicitly for supplied measurement uncertainties. Its mechanism is deliberately deferred until integration. Review existing methods and build their benchmark without assuming that mechanism or requesting its disclosure now.

## Collaboration and pace

There is no rush. A run here means one research or development assignment. Each run should answer one bounded question or produce one reviewable deliverable; a roadmap stage will usually require several runs.

- Before starting, state the question, intended output, expected scale, and stopping point. The user's scoped request counts as agreement; do not seek repeated confirmation for routine work. A broad request to start this project selects the first-run assignment below.
- Work autonomously within that scope, including necessary reading, implementation, checks, and fixes. Keep the user informed of findings and unresolved issues.
- Return at the agreed stopping point with the artifact, evidence and limitations, decisions needing discussion, and one proposed next step. Wait for agreement on that step before continuing, even within the same stage. A completed task or successful test does not authorize further scope.
- If a task proves substantially larger than expected or reveals a consequential choice, preserve the useful work and return for discussion. Make gaps visible rather than filling them through an unbounded expansion of the run.
- Update `PROGRESS.md` at research/development checkpoints with artifact paths, completed work, decisions and their rationale, open questions, and the proposed next step. Keep the brief and supporting documents consistent with agreed revisions.

Use small, understandable increments. Demonstrate a new method on an existing simple problem before extending it to harder problems. Increase method complexity and problem complexity separately. Reporting should support discussion; start with simple templates and add infrastructure when it serves the agreed work.

## First agentic run: literature orientation

**Question:** Which candidate method families plausibly address uncertainty in the latent functions under supplied measurement errors, and what should we investigate next?

Produce a short, source-supported orientation memo, aiming for roughly 2–4 pages of main narrative, plus a compact evidence table and references. This is a scale guide, not a reason to omit qualifications. Export the memo to HTML and PDF using a simple shared source.

Map the initial candidate families listed below, with a small initial set of verified primary sources. Explain the distinctions supported by those sources: inference target, treatment of measurement uncertainty, function outputs, and relevance to the applications. Mark unexamined families and unresolved claims explicitly. Exhaustive source coverage, rankings, a final shortlist, and a fixed experiment protocol belong to later discussions.

Keep a bibliography and search log with queries, dates, search cutoff, paper versions, reading status, and gaps. These can be simple files alongside the memo. Record the outcome in `PROGRESS.md` and propose one focused follow-up, with questions for the user.

**Stop** when the orientation and its evidence gaps are inspectable and the next investigation can be discussed. Do not extend this assignment to finish the literature review, design the benchmark framework, train models, or prepare cluster jobs. The eventual deep review will accumulate across agreed runs. The experiment runtime ceiling does not set a duration for this literature pass.

## Scientific scope and evaluation — current working agreements

The targets are distributions over underlying unknown functions within the range of the observed inputs. **No extrapolation.** Measurement noise belongs in the observation model; uncertainty in future noisy observations is a distinct supporting diagnostic.

Evaluate and report separately:

1. **Posterior approximation:** agreement with a specified Bayesian posterior. Identify the model, prior, likelihood, target quantity, and whether the reference is analytically exact or numerically approximated. Numerical references require validation; comparisons across different target posteriors must say so.
2. **Practical uncertainty quality:** function accuracy, calibration, informativeness, robustness within the domain, and computational cost. Good performance here does not itself establish posterior fidelity, or vice versa.

The four priority problems are:

| Problem | Observations and known relations | Unknown functions | Primary target |
| --- | --- | --- | --- |
| 1. Simple forward regression | Measurements of $y(x)$ | $y(x)$ | $y(x)$ |
| 2. Multiple forward regression | Measurements of $A(x)$ and $B(x)$, with $B(x)=f[A,x]$ | $A(x)$ and $B(x)$ | $A(x)$ |
| 3. Simple inverse regression | Measurements of $y(x)$, with $y(x)=f[u,x]$ | $u(x)$ and $y(x)$ | $u(x)$ |
| 4. Multiple inverse regression | Measurements of $A(x)$ and $B(x)$, with $A(x)=g[u,x]$ and $B(x)=f[A,x]$ | $u(x)$, $A(x)$, and $B(x)$ | $u(x)$ |

The notation $f[h,x]$ allows dependence on both the function and the input. For illustration only, $B(x)=x[A(x)]^2$ shows explicit dependence on both $A$ and $x$; it is not a proposed benchmark relation. Relations may also involve derivatives, integrals, or solutions of known differential equations. The same applies to $g$. Knowing these relations does not make the functions known. A secondary function determined by a relation inherits uncertainty through the joint inferred quantities and need not be parameterized independently. Neural networks represent unknown functions as appropriate to the formulation.

For a measured quantity $z$, use $\mathcal{D}_z=\{(x_{z,i},z_i,\sigma_{z,i})\}_{i=1}^{N_z}$. Different observables may have different locations and sample counts; record their ranges and the evaluation domain. Initially, inputs are exact and errors are independent Gaussian with supplied, known standard deviations that may vary:

$$
z_i=z(x_{z,i})+\epsilon_{z,i},\qquad \epsilon_{z,i}\sim\mathcal{N}(0,\sigma_{z,i}^2).
$$

Governing equations are known; discovering equations is outside current scope. Their solution can be analytical or numerical. Inverse problems require explicit identifiability, boundary/initial conditions, fixed and inferred quantities, and solver-error checks. Correlated or non-Gaussian errors and uncertain noise levels are possible later extensions for discussion.

## Literature review

Build depth in scientific regression, linked observables, inverse problems, and similar applications. The four problems are priorities, not an exhaustive boundary. Explore more distant applications selectively for transferable ideas and blind spots. Distinguish reviewed methods from implemented methods and limit experimental conclusions to the settings tested.

Initial candidates include independent ensembles, MC dropout, variational inference, Laplace approximations, HMC/NUTS for Bayesian neural networks, and specified randomized or repulsive ensembles. **Consider HMC-BNN both as a comparison method and as a possible numerical reference.** The final shortlist and implementation order will be settled collaboratively during the literature stage, considering relevance, prominence, assumptions, implementations, and cost.

Use surveys for discovery and primary sources for substantive claims. Read the relevant full-text sections and supplements where available; snippets and abstracts cannot establish detailed claims or rankings. Record reading/access status and evidence limitations, DOI/arXiv identifiers, exact versions and publication status, supporting section/equation/figure locations, and code revisions when assessing implementations. Maintain search dates, queries, sources, cutoff, and inclusion/exclusion reasons; refresh the review before the final experiment release.

For each method examined, identify the exact variant, inference target, treatment of supplied measurement errors, prior/regularization, supported outputs, cost, and limitations. Separate latent-function uncertainty from noisy-observation prediction. Record departures from published algorithms when implementing them.

## Benchmark principles

Begin with an analytically tractable validation problem, then a small neural-network comparison, then broader problems. An exact posterior for a tractable model does not validate a different neural network's posterior. Knowing a generating function supplies ground truth for reconstruction and coverage, not an exact posterior. HMC references remain numerical approximations; their agreement with themselves is not accuracy evidence.

Use shared datasets, domains, and available noise information; match target posteriors when claiming posterior fidelity. Separate development/tuning from final evaluation. Fix and version reported protocols, retain failures, and use independent datasets with separately recorded algorithm seeds. Grid points, posterior draws, and chains are not independent dataset replicates.

Assess joint function uncertainty where supported as well as pointwise intervals. Distinguish repeated-data coverage for fixed functions from calibration averaged over prior-drawn functions; neither alone proves posterior accuracy. Report numerical uncertainty, measured cost, and unsupported outputs. The [benchmark notes](docs/BENCHMARK_NOTES.md) describe detailed checks, methodological references, candidate metrics, and proposed experiments. Read them before planning or implementing those comparisons.

## Provenance and reporting

Every substantive literature claim must link to its supporting source and location. Distinguish published findings, reproduced results, and interpretations; preserve gaps, failures, and exclusions.

For reported experiments retain a stable run identifier, configuration, code revision and uncommitted changes, dependency versions, data/generator specification, seeds, hardware, precision, solver settings, runtime, logs, and validation status. Preserve the outputs needed to regenerate metrics and figures, with data/artifact checksums. Link each computed figure and table to its runs, processing code, and checks.

HTML and PDF must share the same core argument, figures, and citations, with readable mathematics and static equivalents for essential interactive content. Provide an accessible narrative and inspectable methodological detail. Document rebuilding reports from saved results separately from rerunning experiments, including resources and numerical tolerances. Check that both formats can be rebuilt and read at report checkpoints; use concise progress notes between them.

## Resources and implementation — current working agreements

The current experiment ceiling is approximately **24–48 hours of cluster runtime**, with allocations and aggregate CPU/GPU hours stated. A possible future week-long allowance is outside the current budget. Start with inexpensive local CPU validation and pilots; use measured timings to size larger work and support restart after interruption.

Agents can run on the laptop or desktop. The user schedules cluster jobs through Slurm from reproducible packages prepared by the agent and returns results for validation. Read [computing resources and execution](docs/RESOURCES.md) when sizing runs or preparing this handoff; it lists the available hardware and the proposed package-budget convention.

Use Python primarily; uv, ruff, and pyrefly are preferences. Document mathematical assumptions and implementation. Keep benchmark interfaces extensible for new methods, deriving initial requirements from established methods and the benchmark. Keep project files, instructions, source records, code, and reports in this repository.

## Roadmap — adjustable outline

Each stage contains bounded assignments and discussion checkpoints under the collaboration rules above:

1. Explore the literature progressively, then agree on the baseline shortlist, implementation order, and first protocol.
2. Validate a tractable regression problem, analytic reference, metrics, and report reproduction on a local CPU.
3. Establish simple neural baselines, then a limited checked numerical reference; measure variability and cost.
4. Add selected methods gradually and expand forward-regression comparisons, including linked observables.
5. Introduce inverse problems from simple relations to known dynamics, checking identifiability and numerical error. Consider PINNs only where justified.
6. Integrate the user's method when they are ready and the benchmark is stable.
7. Refresh sources and freeze an auditable set of experiments and reports supporting the paper.

Method complexity also grows gradually: ordinary fitting and simple ensemble/dropout variants first, then selected approximate-inference variants, then broader HMC-BNN or interacting-ensemble studies. A small HMC reference may enter earlier after the basic baselines work. The literature can revise this order; see the detailed proposals in the [benchmark notes](docs/BENCHMARK_NOTES.md).

Settle remaining choices when their evidence becomes available: shortlist and initial protocol through literature discussions, publication repetitions and precision after pilots, and Slurm submission settings and environment details before cluster work. None blocks the initial orientation. Record the current assignment and proposed next step in `PROGRESS.md`; the roadmap itself is not an instruction to advance.
