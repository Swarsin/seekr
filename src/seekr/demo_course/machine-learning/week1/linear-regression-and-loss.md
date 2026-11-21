# Linear Regression and Loss Functions

Supervised regression predicts a continuous numerical target from a vector of
input features. Given paired examples $(x_i, y_i)$, a linear model computes the
weighted sum $\hat{y}_i = w^\top x_i + b$, where $w$ holds the feature slopes
and $b$ is the intercept offset. Each difference $r_i = \hat{y}_i - y_i$ is a
residual that measures how far the prediction misses the observed label.

## Mean Squared Error

To fit parameters automatically, we define an objective function that assigns
a scalar penalty to bad weights. Mean squared error (MSE) averages the squared
residuals across all $n$ training points:

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Squaring each residual keeps the loss non-negative, penalizes large outliers
much more strongly than small errors, and produces a smooth convex bowl in
parameter space. Because MSE is differentiable everywhere, we can inspect its
slope with respect to every weight.

## Closed Form vs. Gradient Descent

For small datasets with well-conditioned features, setting the gradient to
zero yields the normal equations $w = (X^\top X)^{-1} X^\top y$. Inverting
$X^\top X$ scales cubically with the number of features and breaks down when
columns are collinear or too large to hold in memory.

In practice, machine learning pipelines minimize MSE iteratively with
gradient descent. Starting from initial weights, each step computes the partial
derivatives of the loss and nudges $w$ opposite to the uphill direction.
Because MSE is convex for linear models, a well-tuned step size converges
reliably toward the global minimum without inverting matrices.
