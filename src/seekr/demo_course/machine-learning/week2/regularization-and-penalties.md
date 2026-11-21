# Regularization and Penalty Terms

When a flexible model begins to overfit noisy observations, parameter norms
often grow unusually large to cancel tiny fluctuations between correlated
inputs. Regularization curbs this variance by adding a complexity penalty
$\Omega(w)$ directly to the training loss:

$$J(w) = L_{\text{data}}(w) + \lambda \Omega(w)$$

Here $\lambda \ge 0$ controls penalty strength. When $\lambda = 0$, training
minimizes empirical error alone; as $\lambda$ grows, the optimizer prefers
simpler, flatter functions even at the cost of slightly higher training error.

## Ridge (L2) vs. Lasso (L1)

L2 regularization, known as Ridge regression, penalizes the squared Euclidean
norm $\Omega(w) = \sum_j w_j^2$. Because the penalty grows quadratically, large
weights are shrunk aggressively toward zero without forcing them to vanish
completely. In gradient descent, L2 regularization simply multiplies existing
weights by a small decay factor at every update, earning the name weight decay.

L1 regularization, known as Lasso, penalizes the sum of absolute values
$\Omega(w) = \sum_j |w_j|$. The sharp corners of the L1 diamond constraint
drive irrelevant feature weights to exactly zero, yielding sparse models that
perform automatic feature selection alongside shrinkage.

## Practical Pitfalls

Two rules prevent subtle bugs when applying penalties. First, never penalize
the bias intercept term $b$, since shifting predictions vertically adds no
curvature complexity. Second, always standardize feature scales before
fitting Ridge or Lasso; otherwise a single unit change in a tiny-scale column
incurs an unfair penalty compared with large-scale measurements.
