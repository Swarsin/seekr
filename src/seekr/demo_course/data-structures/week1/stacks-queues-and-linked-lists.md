# Stacks, Queues, and Linked Lists

Linear collections restrict where elements enter and leave to enforce clear
program invariants and predictable performance. Choosing between contiguous
buffers and pointer-linked nodes balances worst-case latency against memory
locality.

## Stacks and Queues

A stack follows last-in, first-out (LIFO) discipline: `push` places an item on
top, `pop` removes the newest item, and `peek` inspects the top without mutating
state. Call frames, undo histories, balanced bracket checks, and iterative
depth-first traversal all rely on stacks.

A queue follows first-in, first-out (FIFO) ordering: `enqueue` appends at the
tail while `dequeue` removes from the head. Breadth-first search, task
schedulers, and streaming pipelines depend on queues. Implementing a queue on a
plain dynamic array makes front removal $O(n)$ because remaining items shift
left; a circular ring buffer tracks `head` and `tail` indices modulo capacity
so both ends stay $O(1)$ amortized.

## Linked Lists vs. Arrays

A singly linked list chains heap-allocated nodes where each node stores a value
and a `next` pointer. Inserting or deleting at a known node takes $O(1)$ pointer
reassignments without shifting neighbors, and doubly linked lists add `prev`
links for $O(1)$ removal from either end or middle.

However, linked nodes lack $O(1)$ random indexing and scatter across heap
addresses, causing frequent CPU cache misses during sequential scans. Modern
systems therefore prefer contiguous arrays or chunked blocks unless intrusive
list splicing is strictly required.
