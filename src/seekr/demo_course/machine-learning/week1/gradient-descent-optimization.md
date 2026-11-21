# Gradient Descent Optimization

Gradient descent is an iterative first-order algorithm for minimizing a
differentiable loss surface $L(\theta)$. At step $t$, we evaluate the gradient
vector $\nabla_\theta L(\theta_t)$, which points in the direction of steepest
ascent, and update parameters in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

The scalar hyperparameter $\eta > 0$ is the learning rate. Selecting $\eta$
dominates training stability: if the step size is tiny, progress crawls across
flat plateaus; if $\eta$ is too large, updates overshoot the valley floor and
the loss oscillates or diverges to infinity.

## Batch, Stochastic, and Mini-Batch Variants

Evaluating the exact gradient over an entire dataset of $n$ rows is batch
gradient descent. Each update is deterministic and smooth, yet every single step
must scan every training point before changing a weight.

Stochastic gradient descent (SGD) approximates the gradient from one randomly
sampled example at a time. Individual updates are noisy, which helps escape
shallow local minima and saddle points, though the trajectory fluctuates around
the optimum. Mini-batch gradient descent strikes the practical compromise used
in modern training: it averages gradients over small slices of 32 to 512
examples so hardware vector units stay saturated while updates happen often.

## Feature Scaling and Convergence

When one input feature spans $[0, 1]$ and another spans $[0, 10000]$, the loss
contours stretch into narrow elliptical ravines. Standardizing features to zero
mean and unit variance reshapes those contours into rounder bowls, allowing
gradient descent to take direct paths toward the minimum with fewer zig-zags
and faster convergence.
