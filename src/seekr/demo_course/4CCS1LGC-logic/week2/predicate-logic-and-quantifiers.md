# Predicate Logic and Quantifiers

Propositional logic treats complete sentences as indivisible atoms, so it cannot
express relationships inside statements such as "every prime greater than two
is odd." First-order predicate logic (FOL) adds domain objects, predicates
$P(x)$, function symbols $f(x)$, and quantifiers over variables.

## Universal and Existential Quantifiers

The universal quantifier $\forall x\, P(x)$ asserts that predicate $P$ holds
for every element in the domain of discourse. The existential quantifier
$\exists x\, P(x)$ asserts that at least one element in the domain satisfies
$P$.

Quantifier order changes meaning drastically. In arithmetic over integers,
$\forall x\, \exists y\, (y > x)$ is true because every integer has a larger
successor, whereas $\exists y\, \forall x\, (y > x)$ falsely claims a single
integer exceeds all others.

## Negating Quantified Statements

De Morgan's laws extend naturally to quantifiers:

- $\neg (\forall x\, P(x)) \equiv \exists x\, \neg P(x)$
- $\neg (\exists x\, P(x)) \equiv \forall x\, \neg P(x)$

Disproving a universal claim therefore requires only a single counterexample,
whereas proving an existential claim succeeds by exhibiting one witness.

## Free Variables, Scope, and Interpretations

An occurrence of variable $x$ inside the scope of $\forall x$ or $\exists x$
is bound; otherwise $x$ is free. A closed formula (sentence) has no free
variables and receives a definite truth value once an interpretation supplies a
non-empty domain and concrete relations for each predicate symbol.
