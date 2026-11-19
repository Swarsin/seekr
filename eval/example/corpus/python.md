# Python asyncio cancellation

A cancelled `asyncio.Task` raises `CancelledError` inside the coroutine at its
next await point. Catch it only when you need cleanup, and re-raise it so the
task still ends in the cancelled state. Do not swallow cancellation with a bare
`except Exception`; in modern Python `CancelledError` inherits from
`BaseException` for exactly that reason.

Use `asyncio.TaskGroup` on Python 3.11 or newer to structure a group of related
tasks. If one child fails, the group cancels the remaining children and raises
an `ExceptionGroup`. Shield truly critical cleanup with `asyncio.shield` only
when you understand that the outer await can still be cancelled.
