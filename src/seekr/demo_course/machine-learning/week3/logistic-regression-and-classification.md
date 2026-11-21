# Logistic Regression and Classification Metrics

Binary classification predicts whether an observation belongs to a positive or
negative class. Rather than regressing directly on $\{0, 1\}$ with unbounded
linear outputs, logistic regression maps the linear score $z = w^\top x + b$
through the sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Because $\sigma(z)$ stays strictly between $0$ and $1$, it behaves as a
calibrated probability estimate $\hat{p} = P(y = 1 \mid x)$. Values near zero
correspond to strongly negative predictions, and values near one indicate high
confidence in the positive label.

## Binary Cross-Entropy Loss

Mean squared error creates non-convex flat regions when composed with the
sigmoid curve. Instead, logistic regression minimizes binary cross-entropy
(also called log loss), derived from maximum likelihood estimation:

$$L(y, \hat{p}) = -\big(y \log \hat{p} + (1 - y) \log(1 - \hat{p})\big)$$

When the true label is $y = 1$ and the model predicts $\hat{p} \approx 0.01$,
log loss applies a steep logarithmic penalty, pushing gradients strongly in the
right direction.

## Thresholds, Precision, and Recall

Converting probabilities into discrete labels requires a decision threshold,
commonly $\tau = 0.5$, though asymmetric costs often shift it. Precision
measures the fraction of positive alarms that are truly positive, while recall
measures the fraction of actual positive cases successfully caught.

Sweeping $\tau$ across $[0, 1]$ traces the ROC curve (true positive rate
against false positive rate). For imbalanced datasets where negatives vastly
outnumber positives, precision-recall curves expose false-alarm explosions more
clearly than accuracy or ROC-AUC alone.
