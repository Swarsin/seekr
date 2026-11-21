# Asymptotic Complexity and Dynamic Arrays

Algorithm analysis estimates how runtime and memory scale as input size $n$
grows large, ignoring hardware clock differences and constant factors. Big-O
notation $O(f(n))$ expresses an asymptotic upper bound, $\Omega(f(n))$ marks a
lower bound, and $\Theta(f(n))$ characterizes tight growth from both sides.

## Comparing Common Growth Rates

Constant-time $O(1)$ operations execute in fixed steps regardless of $n$, as in
reading an array slot by index. Logarithmic $O(\log n)$ algorithms halve the
search space per step; binary search on a sorted array locates any target in
roughly $\log_2 n$ comparisons instead of scanning linearly. Linear $O(n)$
passes inspect each item once, whereas quadratic $O(n^2)$ nested loops rapidly
stall once inputs reach tens of thousands of records.

## Dynamic Arrays and Amortized Doubling

Contiguous arrays offer instant random indexing and strong CPU cache locality,
yet their fixed allocation size cannot grow in place. Dynamic arrays (such as
Python `list` or C++ `std::vector`) solve this by allocating a larger buffer
whenever capacity fills and copying existing elements across.

Growing capacity by one element on every append triggers $O(n^2)$ total copies
across $n$ insertions. Instead, doubling capacity on overflow means resizing
happens only after $1, 2, 4, 8, \dots$ appends. The total copy work across $n$
pushes sums to a geometric series bounded by $2n$, giving $O(1)$ amortized
append cost alongside $O(n)$ worst-case single-resize latency.
