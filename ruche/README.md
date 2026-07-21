# Ruche-specific examples

The shell scripts in this directory are historical SLURM launch examples for
the Ruche cluster. They are **not required** for preprocessing, training,
inference, evaluation, or adapter loading on another system.

Before reuse, review every `#SBATCH` directive, module name, account/partition,
repository path, dataset path, checkpoint path, and job dependency. Prefer the
portable commands and `${STHELAR_ROOT}` / `${DATA_ROOT}` placeholders documented
in the root README. Files under `legacy/` are retained only as provenance and
should not be treated as maintained workflows.
