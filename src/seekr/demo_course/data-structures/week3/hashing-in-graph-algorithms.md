# Hashing in Graph Algorithms and Distributed State

Hashing powers far more than standalone key-value dictionaries. In graph
search, streaming deduplication, and partitioned storage systems, fast hash
lookups prevent redundant work and keep state compact.

## Visited Hash Sets for Large State Spaces

Classical breadth-first and depth-first traversals track visited vertices in a
dense array indexed by integer node IDs. Real-world search graphs—such as puzzle
boards, compiler AST states, or web crawl URLs—rarely arrive numbered from 0 to
n - 1. Storing serialized node signatures inside a hash set lets graph algorithms
test whether a state has already been explored in expected O(1) time without
pre-allocating an array across the entire universe of possible states.

When hash collisions occur in high-fanout graph exploration, open addressing
with a moderate load factor keeps pointer chasing minimal and cache lines warm.

## Bloom Filters and Approximate Membership

When visited sets grow too large for RAM, a Bloom filter trades a small tunable
false-positive rate for dramatic memory savings. Using k independent hash
functions over a compact bit array, inserting a key sets k bits to 1, and a
membership query checks whether all k bits are set. Bloom filters never report
false negatives, making them useful front-end guards before disk lookups.

## Consistent Hashing for Partitioning

Distributed caches and partitioned graph stores cannot afford modulo rehashing
when servers join or leave, because changing table size m relocates nearly
every key. Consistent hashing places both node tokens and hashed keys onto a
shared ring so adding or removing one shard moves only the fraction of keys
adjacent to that shard.
