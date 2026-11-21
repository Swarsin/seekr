# Hash Tables and Collision Resolution

A hash table stores key-value entries with expected $O(1)$ lookup, insertion,
and deletion. Instead of comparing keys in sorted order, hashing maps an
arbitrary key $k$ through a deterministic hash function $h(k)$ into an array
bucket index `h(k) mod m` inside a table of capacity $m$.

## Good Hash Functions and Load Factor

A strong hash function distributes keys uniformly across buckets so small
changes in the input bits flip roughly half the output bits (the avalanche
effect). The load factor $\alpha = n / m$ measures how full the table is with
$n$ stored entries across $m$ slots. Keeping $\alpha$ bounded below a resize
threshold prevents chains or probe runs from growing linearly.

## Handling Hash Collisions

By the pigeonhole principle, distinct keys inevitably map to the same bucket
index, causing a hash collision. Two classic strategies resolve collisions:

- **Separate chaining:** Each bucket stores a small linked list or dynamic
  bucket of all entries hashing to that slot. Lookup scans only the short list
  at `h(k) mod m`, staying $O(1 + \alpha)$ on average.
- **Open addressing:** All entries live directly in the main backing array. On
  collision, probing follows a deterministic sequence—such as linear probing
  $(h(k) + i) \bmod m$ or double hashing—until an empty slot appears.

Linear probing enjoys excellent cache locality, though clusters form when
$\alpha$ exceeds roughly $0.7$, forcing a table rehash into a larger prime or
power-of-two capacity.
