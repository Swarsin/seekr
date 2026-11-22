Output "docs/img/demo.gif"

Set Shell "bash"
Set FontSize 14
Set Width 1100
Set Height 660
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type@0ms 'unset SEEKR_CONFIG'
Enter
Type@0ms 'export PATH="$PWD/.venv/bin:$PATH" PS1="$ "'
Enter
Type@0ms 'DEMO_TMP=$(mktemp -d)'
Enter
Type@0ms 'export XDG_CONFIG_HOME="$DEMO_TMP/config" XDG_DATA_HOME="$DEMO_TMP/data" XDG_RUNTIME_DIR="$DEMO_TMP/run"'
Enter
Type@0ms 'mkdir -p "$XDG_CONFIG_HOME/seekr" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR"'
Enter
Type@0ms 'clear'
Enter
Wait+Line@10s /^\$$/
Sleep 300ms
Show

Type "seekr init --demo --yes"
Enter
Wait+Line@30s /^\$$/
Sleep 1500ms

Type 'seekr search "bias variance" --module machine-learning --week 2 -k 2'
Enter
Wait+Line@20s /^\$$/
Sleep 2000ms

Type 'seekr search "trees" --spread -k 2'
Enter
Wait+Line@20s /^\$$/
Sleep 2500ms

Hide
Type@0ms 'rm -rf "$DEMO_TMP"'
Enter
Wait+Line@5s /^\$$/
