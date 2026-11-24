# Figure Input Contracts

Each figure script must document its figure identifier, upstream workflow, required input schema, sample-selection rule, statistical transformations, and output files.

The permitted data path is:

```text
Immutable source records and model predictions
    -> Explicit sample selection
    -> Statistical calculation
    -> Figure generation
```

Subgroup and figure code must not alter clinical fields, model scores, labels, identifiers, or cohort membership. Ad hoc record-modification outputs are not valid figure inputs.

AUC values, confidence intervals, and other displayed performance statistics must be computed from the same input records used for the figure. Manual text must not be the sole source of a displayed numerical result.

All scripts will run from the repository root and accept input and output locations through configuration or command-line arguments.
