# Strong Induction, Structural Induction, and Soundness

Weak induction assumes only the single predecessor P(k) when proving P(k + 1).
Many theorems in logic and discrete mathematics depend on arbitrary smaller
subproblems rather than just k, calling for strong or structural forms of
proof by induction.

## Strong Induction

In strong induction (also called complete induction), the inductive hypothesis
assumes P(j) holds for all integers from the base n_0 up through k simultaneously,
and uses any subset of those earlier cases to establish P(k + 1). Proving that
every integer n >= 2 factors into primes relies on strong induction: when
n = a * b is composite, both factors satisfy 2 <= a, b < n rather than equaling
n - 1.

## Structural Induction on Syntax Trees

Logical formulas, abstract syntax trees, and linked lists are defined
inductively from atomic leaves and constructor rules rather than linear integer
counts. Structural proof by induction mirrors the grammar directly:

1. **Base cases:** Verify the property for every atomic leaf, such as variables
   or Boolean constants.
2. **Inductive cases:** For each compound constructor (negation, conjunction,
   implication), assume the property holds on every immediate subformula and
   prove it holds for the combined expression.

For instance, structural induction proves that every well-formed propositional
formula contains equal numbers of left and right parentheses, or that replacing
an equivalent subformula preserves overall truth value.

## Soundness and Completeness Recap

In deductive systems, soundness guarantees that every syntactically derivable
theorem is semantically valid (no false proofs exist). Proving soundness uses
induction on proof length: axioms are valid base cases, and each inference
rule preserves validity across steps. Completeness establishes the converse:
every semantically valid formula admits a finite derivation.
