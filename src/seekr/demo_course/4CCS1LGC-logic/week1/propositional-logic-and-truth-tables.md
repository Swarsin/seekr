# Propositional Logic and Truth Tables

Propositional logic formalizes reasoning over declarative statements—atomic
propositions such as $P$ and $Q$—that take exactly one truth value: true
($\top$) or false ($\bot$). Compound formulas combine atoms using negation
($\neg$), conjunction ($\land$), disjunction ($\lor$), implication ($\to$), and
biconditional ($\leftrightarrow$).

## Implication, Equivalence, and Normal Forms

Material implication $P \to Q$ evaluates to false only when $P$ is true and $Q$
is false; whenever the antecedent $P$ is false, the implication holds vacuously.
Two formulas are logically equivalent when every truth assignment gives them
identical truth values.

Useful identities rewrite implications into disjunctions via $P \to Q \equiv
\neg P \lor Q$ and push negations inward using De Morgan's laws:
$\neg(P \land Q) \equiv \neg P \lor \neg Q$ and $\neg(P \lor Q) \equiv
\neg P \land \neg Q$. Repeatedly distributing $\lor$ over $\land$ places any
formula into Conjunctive Normal Form (CNF), a conjunction of disjunctive
clauses suited for SAT solvers and resolution refutation.

## Tautologies, Satisfiability, and Inference

A formula is a tautology (valid) if every row of its truth table evaluates to
true, satisfiable if at least one row is true, and unsatisfiable (a
contradiction) if no assignment satisfies it. Checking whether premises
$\Gamma$ semantically entail conclusion $\varphi$ reduces to testing whether
$\Gamma \land \neg \varphi$ is unsatisfiable.

Sound inference rules such as modus ponens (from $P$ and $P \to Q$, infer $Q$)
and propositional resolution derive valid consequences syntactically without
constructing exponential truth tables by hand.
