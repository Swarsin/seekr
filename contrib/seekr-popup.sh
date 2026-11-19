#!/usr/bin/env bash
# seekr rofi popup: ask for a query, choose one hit, open it at its location.
#
# Requires rofi and seekr on PATH.  Automated tests only check this script's
# shell syntax and wiring; the interactive rofi/zathura/editor flow has been
# verified manually only.  Cancelling either rofi prompt exits 0.
set -u
set -o pipefail

if ! query="$(rofi -dmenu -p seekr)"; then
    exit 0
fi
if [[ -z "$query" ]]; then
    exit 0
fi

hits="$(seekr pick "$query")"
pick_status=$?
if (( pick_status != 0 )); then
    printf 'seekr popup: seekr pick failed (exit %d)\n' "$pick_status" >&2
    exit "$pick_status"
fi
if [[ -z "$hits" ]]; then
    exit 0
fi

if ! selection="$(rofi -dmenu -p seekr <<<"$hits")"; then
    exit 0
fi
if [[ -z "$selection" ]]; then
    exit 0
fi

# The pick line is one argv element; `--` keeps a relative leading-hyphen
# path from being parsed as an option.  It is never interpreted by a shell.
exec seekr open -- "$selection"
