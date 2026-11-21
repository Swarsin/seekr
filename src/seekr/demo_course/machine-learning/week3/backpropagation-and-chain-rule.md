# Backpropagation and the Chain Rule

Training multi-layer neural networks requires computing gradients for millions
of weights efficiently. Running finite differences per parameter would demand a
separate forward pass for every weight; backpropagation solves this in a single
backward sweep using reverse-mode automatic differentiation.

## Forward and Backward Passes

A network forms a directed acyclic graph of elementary operations: matrix
multiplications, bias additions, and nonlinear activations such as ReLU or
tanh. During the forward pass, each node computes its output and caches local
intermediate values needed for derivatives.

During the backward pass, we start at the scalar loss with upstream sensitivity
$\frac{\partial L}{\partial L} = 1$ and propagate gradients backward toward
the inputs. Whenever node $z = f(u, v)$ receives upstream signal $g =
\frac{\partial L}{\partial z}$, the multivariate chain rule sends $g \cdot
\frac{\partial z}{\partial u}$ along edge $u$ and $g \cdot \frac{\partial
z}{\partial v}$ along edge $v$, summing contributions where paths merge.

## Adaptive Gradient Descent

Once backpropagation supplies parameter gradients, optimizers refine standard
gradient descent to handle deep non-convex landscapes:

- **Momentum** maintains an exponential moving average of past gradients,
  damping sideways oscillations across steep canyon walls while accelerating
  along persistent downhill directions.
- **Adam** combines momentum with per-parameter adaptive scaling using running
  estimates of the first and second gradient moments.

## Vanishing and Exploding Gradients

Multiplying many small local derivatives across deep chains shrinks early-layer
gradients toward zero, stalling learning with saturating sigmoids. ReLU
activations, careful Xavier or Kaiming weight initialization, and residual skip
connections preserve healthy gradient flow through deep architectures.
