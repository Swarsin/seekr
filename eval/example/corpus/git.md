# Git recovery notes

`git reflog` records where branch tips pointed recently, including commits that
are no longer reachable from any branch. Find the lost commit hash, then create
a new branch at it with `git switch -c recovered <hash>`. Do not run garbage
collection until you have recovered what you need.

To undo a public commit safely, use `git revert`; it creates a new commit that
inverts the old change. `git reset --hard` rewrites your current branch and can
discard uncommitted work, so use it only after checking `git status` and saving
anything important. A conflicted rebase can be aborted with `git rebase --abort`.
