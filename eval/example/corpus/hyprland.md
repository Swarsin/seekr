# Hyprland popup launcher

The seekr popup uses rofi twice: the first dmenu prompt reads the search query,
and the second chooses among the result lines. A Hyprland `bind` runs the shell
script without opening a terminal. Bind a spare key to the script and keep the
script on `PATH`, or use its absolute path.

The popup delegates search to `seekr pick` and opening to `seekr open`, so the
same index and configuration as the command line are used. Cancel either rofi
prompt with Escape. Pin the popup to a floating window rule if your compositor
places it off-screen.
