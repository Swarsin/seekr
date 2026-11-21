# Proof by Induction: Foundations

A proof by induction establishes that a predicate $P(n)$ holds for every
natural number $n \ge n_0$ without checking infinitely many cases individually.
Think of mathematical induction as knocking down an infinite line of dominoes:
if the first tile falls and every falling tile tips its successor, all tiles
must fall.

## Structure of Weak Induction

Every proof by induction consists of three explicit parts:

1. **Base case:** Prove directly that $P(n_0)$ is true for the starting
   index (commonly $n_0 = 0$ or $n_0 = 1$).
2. **Inductive hypothesis:** Fix an arbitrary integer $k \ge n_0$ and assume
   that $P(k)$ holds.
3. **Inductive step:** Using the inductive hypothesis $P(k)$, prove that
   $P(k + 1)$ must also hold.

Omitting or mishandling the base case invalidates the entire proof: the fake
claim that $n = n + 1$ has a valid inductive step if $k = k + 1$ implies
$k + 1 = k + 2$, yet fails completely at the base case.

## Proving Summations and Recurrences

Induction excels on closed-form series and algorithm recurrences. To show that
$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$, the base case checks $n = 1$ gives $1$.
Assuming the formula at $k$, adding the next term $(k + 1)$ factors cleanly as
$\frac{k(k+1)}{2} + (k + 1) = \frac{(k+1)(k+2)}{2}$, closing the inductive step.

Similarly, proving an upper bound $T(n) \le c n \log n$ for merge sort or
showing that a binary tree of height $h$ has at most $2^{h+1} - 1$ nodes turns
recursive definitions directly into algebraic steps inside the inductive
argument.
