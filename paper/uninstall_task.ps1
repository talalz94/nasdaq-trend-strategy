<#
Removes the scheduled tasks created by install_task.ps1. Your paper account files (paper\data) are kept.
    powershell -ExecutionPolicy Bypass -File paper\uninstall_task.ps1
#>
foreach ($name in @("NDX-Trend-Paper-Tick", "NDX-Trend-Paper-Watchdog", "NDX-Trend-Paper-Wake")) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "removed $name"
    } else {
        Write-Host "$name was not installed"
    }
}
