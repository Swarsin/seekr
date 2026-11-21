# Model Evaluation and Cross-Validation

Measuring accuracy on the same examples used for fitting rewards memorization
and hides generalization failures. Reliable evaluation isolates training,
hyperparameter selection, and final reporting into strictly separated data
partitions.

## Train, Validation, and Test Splits

A three-way split assigns distinct responsibilities:

- **Training split:** Fits model parameters such as regression weights or tree
  thresholds.
- **Validation split:** Compares candidate architectures, regularization
  strengths, and early stopping points.
- **Test split:** Touched exactly once after all design decisions freeze to
  produce an unbiased estimate of deployment performance.

Repeatedly tuning knobs against the test split leaks information across runs
and turns the test benchmark into an optimistic validation set.

## K-Fold Cross-Validation

When datasets are small, a single random validation slice exhibits high
variance depending on which examples happened to land in it. K-fold
cross-validation partitions the development set into $k$ equal folds (often
$k = 5$ or $k = 10$). Each fold takes a turn serving as validation while the
remaining $k - 1$ folds train a fresh model, and scores are averaged across all
runs.

For classification with skewed class proportions, stratified k-fold preserves
the exact minority-class ratio inside every fold.

## Preventing Data Leakage

Data leakage happens whenever statistics from held-out rows contaminate
training transforms. Computing standardization means, TF-IDF vocabularies, or
imputation medians across the full dataset before splitting inflates cross-
validation scores artificially. Every preprocessing step must fit exclusively
inside each training fold and only transform validation or test rows.
