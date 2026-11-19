# SQLite local search notes

SQLite in WAL mode allows readers and one writer to proceed concurrently.
Run `PRAGMA journal_mode=WAL` once and keep the `-wal` and `-shm` files beside
the database. A checkpoint copies committed WAL frames back into the main
database; automatic checkpoints normally keep the WAL from growing forever.
Never copy only the main database file while a writer is active.

FTS5 provides ranked full-text matching with the porter unicode61 tokenizer.
The sqlite-vec extension adds nearest-neighbour vector search inside the same
transactional store. Wrap retrieval and chunk fetching in one read transaction
so a concurrent reindex cannot change the ranking halfway through.
