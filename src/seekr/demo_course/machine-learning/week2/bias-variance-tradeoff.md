# The Bias-Variance Tradeoff and Overfitting

A supervised model can achieve zero error on its training set yet fail badly on
unseen examples. Understanding why a model overfits requires decomposing
expected prediction error into irreducible noise, squared bias, and variance.

## Bias vs. Variance

Bias measures systematic error caused by overly rigid modeling assumptions.
Fitting a straight line to curved quadratic data produces high bias: the model
misses the underlying signal in both training and test splits, a failure mode
known as underfitting.

Variance measures how sharply a fitted model changes when trained on a
different random draw of the same dataset. A high-degree polynomial or deep
unconstrained decision tree has low bias and high capacity, so it bends to
memorize random measurement noise alongside the true relationship. Because
that noise does not repeat in fresh data, test error climbs rapidly while
training error drops near zero.

## Diagnosing Overfitting

Why does a model overfit in practice? Three triggers recur across projects:
excessive model capacity relative to sample size, training for too many
epochs without early stopping, and unpenalized weights that amplify spurious
feature correlations.

Comparing learning curves separates bias from variance immediately:

- **High bias (underfitting):** Both training loss and validation loss plateau
  at an unacceptably high value; gathering more rows rarely helps until model
  flexibility increases.
- **High variance (overfitting):** Training loss stays very low while
  validation loss rises or remains far above training error, leaving a wide
  generalization gap.

To balance the bias-variance tradeoff, engineers simplify features, collect
more diverse training examples, add regularization penalties, or select model
complexity using held-out validation curves.
