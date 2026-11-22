Output "docs/img/demo.gif"

Set Shell "bash"
Set FontSize 15
Set Width 1050
Set Height 620
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type@0ms 'export PATH="$PWD/.venv/bin:$PATH" PS1="$ "'
Enter
Type@0ms 'DEMO_TMP=$(mktemp -d)'
Enter
Type@0ms 'export XDG_CONFIG_HOME="$DEMO_TMP/config" XDG_DATA_HOME="$DEMO_TMP/data" XDG_RUNTIME_DIR="$DEMO_TMP/run"'
Enter
Type@0ms 'mkdir -p "$XDG_CONFIG_HOME/seekr" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR"'
Enter
Type@0ms 'printf "roots = [\"%s/src/seekr/demo_course\"]\n" "$PWD" > "$XDG_CONFIG_HOME/seekr/config.toml"'
Enter
Type@0ms 'clear'
Enter
Wait+Line@10s /^\$$/
Sleep 300ms
Show

Type "seekr index"
Enter
Wait+Line@20s /^\$$/
Sleep 1s

Type 'seekr search "why does my model overfit" -k 2'
Enter
Wait+Line@20s /^\$$/
Sleep 1500ms

Type 'seekr search "modus ponens" --mode keyword -k 2'
Enter
Wait+Line@15s /^\$$/
Sleep 2500ms

Hide
Type@0ms 'rm -rf "$DEMO_TMP"'
Enter
Wait+Line@5s /^\$$/
