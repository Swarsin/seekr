# Graph Representations and Traversals

Graphs model pairwise relationships among vertices $V$ connected by directed or
undirected edges $E$. Choosing between an adjacency matrix and an adjacency list
shapes both memory footprint and traversal speed.

## Adjacency Lists vs. Matrices

An adjacency matrix uses a $|V| \times |V|$ boolean or weight table. Edge
existence queries finish in $O(1)$, yet storing all pairs consumes
$\Theta(|V|^2)$ memory even when edges are sparse. An adjacency list stores one
neighbor sequence per vertex, requiring $\Theta(|V| + |E|)$ space and iterating
only over actual outgoing edges—ideal for road networks, dependency graphs, and
citation webs.

## Breadth-First and Depth-First Search

Breadth-first search (BFS) explores vertices in concentric layers of increasing
hop count from a start node using a FIFO queue. Marking nodes as visited upon
enqueue ensures each vertex and edge is inspected once in $O(|V| + |E|)$ time,
and the first path reaching any target in an unweighted graph is guaranteed to
have minimum hop length.

Depth-first search (DFS) plunges down unvisited branches as far as possible via
recursion or an explicit LIFO stack before backtracking. Coloring vertices as
unvisited, active, and finished detects directed cycles cleanly when an edge
points to an active ancestor. In directed acyclic graphs (DAGs), reversing the
post-order finish sequence yields a valid topological ordering of tasks.
