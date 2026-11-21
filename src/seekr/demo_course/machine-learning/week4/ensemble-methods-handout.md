# Ensemble Methods: Bagging and Boosting

Individual decision trees are fast and interpretable, yet a single deep tree
is notoriously unstable: flipping a few training points can rearrange top
splits completely. Ensemble learning resolves this weakness by combining many
base learners into a single stronger predictor, directly targeting the
bias-variance tradeoff.

## Bagging and Random Forests

Bootstrap aggregating (bagging) trains B independent models in parallel on
separate bootstrap samples drawn with replacement from the training set, then
averages predictions or takes a majority vote. Because averaging independent
noisy estimators shrinks variance by roughly 1 / B, bagging dramatically
reduces overfitting without increasing bias.

Random forests extend bagging for decision trees by sampling a random subset
of m candidate features at each split instead of scanning every column. That
extra randomness decorrelates trees grown on datasets with one dominant
predictor, tightening variance reduction even further.

## Gradient Boosting

Boosting builds base learners sequentially rather than in parallel. Each new
shallow tree fits the remaining errors—or negative gradients of the loss—left
behind by the current ensemble. Because shallow trees have high bias when used
alone, sequential additive fitting systematically drives down bias while
keeping tree depth small to contain variance.

## Choosing Between Variance and Bias Control

Comparing ensemble families clarifies how bias-variance behavior governs
selection:

- **Bagging and random forests** excel when base learners already have low
  bias and high variance, smoothing jagged boundaries cleanly.
- **Boosting with small learning rates and early stopping** shines on tabular
  problems where underfitting shallow learners must gradually capture subtle
  interactions without memorizing outliers.
