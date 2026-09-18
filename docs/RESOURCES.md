# Computing resources and execution

Read when sizing experiments or preparing execution. [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) governs scope and the pace of work.

## Available hardware

| Environment | Resources useful for planning | Execution |
| --- | --- | --- |
| Laptop | AMD Ryzen AI 7 445, 6 cores / 12 threads; approximately 30 GiB RAM | Agent and local runs |
| Desktop | AMD Ryzen 5 5600G, 6 cores / 12 threads; approximately 14 GiB RAM | Agent and local runs |
| Cluster | H200 GPUs, Sapphire Rapids CPUs, nodes with up to 2000 GB RAM | User-submitted Slurm jobs; no agent on the cluster |

Both local machines run Fedora Linux and have integrated AMD graphics without a discrete GPU. Use CPU execution for initial validation and small pilots. Check available memory and disk space on the execution machine before substantial runs.

## Runtime and allocation

The current ceiling is **24–48 hours of cluster runtime per experiment**. Keep development runs inexpensive, use pilot timings to size larger work, and support restart after interruption. Specify allocated resources and aggregate CPU/GPU hours as well as elapsed runtime. Changes to the agreed allowance require discussion.

**Proposed budget convention:** the ceiling covers the complete, predeclared experiment package, including tuning, repetitions, and reference calculations. Splitting it into jobs does not reset the allowance. Record queue waiting separately. Confirm package scope and allocation before scheduling a large run.

## Cluster handoff

The agent prepares a reproducible package with the code revision, environment setup, inputs/configuration, exact commands, a Slurm submission script, requested resources, a runtime estimate based on pilot timings, restart instructions, and expected outputs. Jobs must run unattended. The user schedules them and returns results and logs for validation.

Confirm cluster-specific Slurm settings, software environment, resource allocations, and input/result transfer when preparing the first cluster package.
