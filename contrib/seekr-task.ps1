<#
Register a per-user Task Scheduler task that starts the seekr service at logon.

  powershell -ExecutionPolicy Bypass -File contrib\seekr-task.ps1
  powershell -ExecutionPolicy Bypass -File contrib\seekr-task.ps1 -SeekrPath C:\path\to\seekr.exe

Remove it with:
  Unregister-ScheduledTask -TaskName "seekr service" -Confirm:$false
#>
param(
    [string]$SeekrPath = (Get-Command seekr -ErrorAction SilentlyContinue).Source
)

if (-not $SeekrPath) {
    throw "seekr was not found on PATH; pass -SeekrPath C:\path\to\seekr.exe"
}

$action = New-ScheduledTaskAction -Execute $SeekrPath -Argument "service start --port 8000"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# The service runs until stopped, so remove the default 72-hour time limit.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName "seekr service" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Starts the seekr search service at logon" -Force
