# Binary Search Trees and Heaps

Tree structures organize ordered data hierarchically so searches, insertions,
and priority extractions remain logarithmic even under continuous updates.

## Binary Search Trees and Balance

A binary search tree (BST) stores keys with a strict structural invariant: for
every node $v$, all keys in $v$'s left subtree are strictly smaller than
$v.\text{key}$, and all keys in its right subtree are strictly larger. An
in-order traversal therefore visits keys in sorted ascending order.

Search, insertion, and deletion walk down a single root-to-leaf path in
$O(h)$ time, where $h$ is tree height. Inserting already-sorted keys into an
unbalanced BST degenerates the tree into a skewed chain of height $O(n)$.
Self-balancing trees such as AVL trees and red-black trees repair skew after
updates using local tree rotations, guaranteeing $h = O(\log n)$.

## Binary Min-Heaps and Priority Queues

A binary min-heap is a complete binary tree packed densely into a flat array,
where parent $i$ has children at $2i + 1$ and $2i + 2$. Unlike a BST, a min-heap
enforces only the vertical heap invariant: every parent key is less than or
equal to its children's keys.

Peeking the global minimum takes $O(1)$ at index $0$. Inserting pushes at the
end and sifts upward in $O(\log n)$, while extracting the minimum swaps the last
leaf to the root and sifts downward in $O(\log n)$. Building a heap from an
unsorted array bottom-up with Floyd's `heapify` runs in tight $O(n)$ time.
