<#
Dead-man's switch that does not depend on Python: if the runner's heartbeat is older than -MaxAgeMinutes,
write an alert and show a desktop notification. It stays quiet for the first 20 minutes after Windows
starts, wakes from sleep or modern standby, or resumes from hibernation (Fast Startup included), so a PC
that was simply off or asleep does not raise a false alarm before the runner has caught up.
-DataDir is passed by install_task.ps1 (the runner's data folder, paths.data_dir in config.toml).
#>
param([int]$MaxAgeMinutes = 60, [string]$DataDir = "")

$data = if ($DataDir) { $DataDir } else { Join-Path $PSScriptRoot "data" }
$hb = Join-Path $data "heartbeat.json"

$msg = $null
if (-not (Test-Path $hb)) {
    if (Test-Path (Join-Path $data "state.json")) {
        $msg = "The paper runner has never completed a run. Check the NDX-Trend-Paper-Tick scheduled task."
    } else { exit 0 }                       # no account yet: nothing to watch
} else {
    try {
        $j = Get-Content $hb -Raw | ConvertFrom-Json
        $last = [DateTimeOffset]::Parse($j.last_tick_utc).UtcDateTime
        $age = ((Get-Date).ToUniversalTime() - $last).TotalMinutes
        if ($age -gt $MaxAgeMinutes) {
            $msg = "The paper runner has not run for {0:N0} minutes (last run {1:u}). Check the NDX-Trend-Paper-Tick task, Python and {2}." -f $age, $last, (Join-Path $data "runner.log")
        }
    } catch {
        $msg = "Could not read ${hb}: $($_.Exception.Message)"
    }
}
if (-not $msg) { exit 0 }                    # the normal case: a recent heartbeat, nothing else is checked

# stale: stay quiet if Windows only just started or woke up (the runner needs a few minutes to catch up)
$since = (Get-Date).AddMinutes(-20)
try { if ((Get-CimInstance Win32_OperatingSystem).LastBootUpTime -gt $since) { exit 0 } } catch { }
foreach ($f in @(@{ LogName = "System"; ProviderName = "Microsoft-Windows-Power-Troubleshooter"; Id = 1; StartTime = $since },
                 @{ LogName = "System"; ProviderName = "Microsoft-Windows-Kernel-Boot"; Id = 27; StartTime = $since },
                 @{ LogName = "System"; ProviderName = "Microsoft-Windows-Kernel-Power"; Id = 507; StartTime = $since })) {
    try { if (Get-WinEvent -FilterHashtable $f -MaxEvents 1 -ErrorAction Stop) { exit 0 } } catch { }
}

$line = "{0:yyyy-MM-dd HH:mm:ss}Z [WARN] watchdog | {1}" -f (Get-Date).ToUniversalTime(), $msg
New-Item -ItemType Directory -Force -Path $data | Out-Null
try { Add-Content -Path (Join-Path $data "alerts.log") -Value $line } catch { }
try { Add-Content -Path (Join-Path $data "ALERT_ACTIVE.txt") -Value $line } catch { }
try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
    $t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $x = $t.GetElementsByTagName("text")
    $x.Item(0).AppendChild($t.CreateTextNode("NDX trend paper: runner stopped")) > $null
    $x.Item(1).AppendChild($t.CreateTextNode($msg)) > $null
    $app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
} catch { }
