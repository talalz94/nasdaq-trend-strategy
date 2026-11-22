<#
Registers Windows Task Scheduler tasks for the paper-trading runner. They run as you, only while you are
logged on, so Windows stores no password and no new credentials are created.

  NDX-Trend-Paper-Tick      every 5 minutes, at log-on, and one minute after the PC wakes from sleep, modern
                            standby or hibernation; a run missed while the PC was off starts as soon as it is back.
                            Runs: pythonw paper\runner.py tick (no console window). When nothing is due - almost
                            always - a run takes a fraction of a second and loads nothing heavy.
  NDX-Trend-Paper-Watchdog  hourly: paper\watchdog.ps1 - a desktop alert if the runner has stopped running.
                            It does not use Python, so it still works if Python or the runner breaks.
  NDX-Trend-Paper-Wake      only with -WakeToRun: wakes a sleeping PC one minute into each weekly decision
                            window (Friday 15:41 New York time, or the week's last trading day; 12:41 on 1 pm
                            half-days) - once a week - for the exact Friday-close trade. It covers the next ~10
                            months (Task Scheduler allows 48 triggers per task): re-run this script to extend it.
                            Windows honours it only if "Allow wake timers" is enabled in your power plan; this
                            script never changes power settings, it only reads them.

The PC does NOT have to be on at any particular time. If it is off or asleep in the Friday window (15:40-15:48
New York time = Saturday 01:40, or 00:40 in US summer time, in Pakistan), the decision is made as soon as the PC
is on again, from Friday's official close, and trades at the next close it can still reach (catch-up).

Run once (from the project folder):
    powershell -ExecutionPolicy Bypass -File paper\install_task.ps1 [-WakeToRun] [-EveryMinutes 5]
-EveryMinutes (1-8, default 5) must not exceed the 8-minute window or a window could pass with no run.
Remove with paper\uninstall_task.ps1. Re-running this script updates the tasks in place.
#>
param(
    [ValidateRange(1, 8)][int]$EveryMinutes = 5,
    [switch]$WakeToRun
)
$ErrorActionPreference = "Stop"

$paper = $PSScriptRoot
$root = Split-Path -Parent $paper
$python = (Get-Command python -ErrorAction Stop).Source
$pythonw = Join-Path (Split-Path -Parent $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = $python }

# the runner's own settings (data folder, half-days); a broken paper\config.toml stops the install here
$infoJson = & $python "$paper\runner.py" install-info
if ($LASTEXITCODE -ne 0) { throw "python paper\runner.py install-info failed: fix the error shown above (paper\config.toml?) and re-run" }
$info = ($infoJson -join "") | ConvertFrom-Json
$dataDir = $info.data_dir
if (-not $info.state_exists) {
    Write-Warning "No paper account yet. Run first:  python paper\runner.py init --cash 10000"
}
if ($EveryMinutes -gt $info.window_minutes) {
    throw "-EveryMinutes ($EveryMinutes) must not exceed the decision window ($($info.window_minutes) minutes)"
}

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$tickAction = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$paper\runner.py`" tick" -WorkingDirectory $root

function New-RepeatingDailyTrigger([string]$At, [timespan]$Every) {
    $t = New-ScheduledTaskTrigger -Daily -At $At
    $rep = (New-ScheduledTaskTrigger -Once -At $At -RepetitionInterval $Every -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
    $rep.StopAtDurationEnd = $false          # never kill a run that is still going when the day's repetition ends
    $t.Repetition = $rep
    return $t
}

function New-ResumeTrigger {
    # System log: Power-Troubleshooter 1 = back from sleep or hibernation; Kernel-Power 507 = leaving modern
    # standby (S0 low-power idle laptops log this instead of event 1)
    $class = Get-CimClass -Namespace Root/Microsoft/Windows/TaskScheduler -ClassName MSFT_TaskEventTrigger
    $t = New-CimInstance -CimClass $class -ClientOnly
    $t.Enabled = $true
    $t.Delay = "PT1M"                        # give Wi-Fi a minute to reconnect
    $t.Subscription = '<QueryList><Query Id="0" Path="System"><Select Path="System">' +
        "*[System[(Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1) or " +
        "(Provider[@Name='Microsoft-Windows-Kernel-Power'] and EventID=507)]]</Select></Query></QueryList>"
    return $t
}

# ---- tick: every N minutes all day, at log-on, after waking; a run missed while off starts when back on
$daily = New-RepeatingDailyTrigger "00:00" (New-TimeSpan -Minutes $EveryMinutes)
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$resume = New-ResumeTrigger
$tickSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "NDX-Trend-Paper-Tick" -Action $tickAction -Trigger @($daily, $logon, $resume) `
    -Principal $principal -Settings $tickSettings -Force `
    -Description "NDX trend strategy paper trading (simulated, no broker): books closes and makes the weekly decision." | Out-Null

# ---- watchdog: hourly, independent of Python
$hourly = New-RepeatingDailyTrigger "00:30" (New-TimeSpan -Hours 1)
$wdAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$paper\watchdog.ps1`" -DataDir `"$dataDir`"" `
    -WorkingDirectory $root
$wdSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "NDX-Trend-Paper-Watchdog" -Action $wdAction -Trigger $hourly `
    -Principal $principal -Settings $wdSettings -Force `
    -Description "Alerts if the NDX paper runner has stopped running." | Out-Null

# ---- the decision windows in this PC's time (New York 15:40 in US winter and US summer time)
$ny = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$lead = [int]$info.signal_minutes_before_close
function Convert-NyToLocal([datetime]$NyTime) { return [TimeZoneInfo]::ConvertTime($NyTime, $ny, [TimeZoneInfo]::Local) }
$winterFri = Convert-NyToLocal ([datetime]"2027-01-15 16:00").AddMinutes(-$lead)
$summerFri = Convert-NyToLocal ([datetime]"2027-07-16 16:00").AddMinutes(-$lead)
$windowText = "{0:dddd HH:mm} (US winter time) / {1:dddd HH:mm} (US summer time) your time" -f $winterFri, $summerFri

# ---- optional wake task: one trigger at the exact instant of each upcoming decision window (UTC, so daylight
# saving in New York or here can never shift it)
if ($WakeToRun) {
    $triggers = @()
    foreach ($iso in $info.wake_times_utc) {
        $utc = [DateTime]::Parse($iso, [Globalization.CultureInfo]::InvariantCulture,
                                 [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal)
        $triggers += New-ScheduledTaskTrigger -Once -At $utc
    }
    $lastWake = [DateTime]::Parse($info.wake_times_utc[-1]).ToLocalTime()
    $wakeSettings = New-ScheduledTaskSettingsSet -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
    Register-ScheduledTask -TaskName "NDX-Trend-Paper-Wake" -Action $tickAction -Trigger $triggers `
        -Principal $principal -Settings $wakeSettings -Force `
        -Description "Wakes the PC for the NDX paper runner's decision windows (needs 'Allow wake timers')." | Out-Null
} elseif (Get-ScheduledTask -TaskName "NDX-Trend-Paper-Wake" -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName "NDX-Trend-Paper-Wake" -Confirm:$false
    Write-Host "removed NDX-Trend-Paper-Wake (re-run with -WakeToRun to keep it)"
}

$extra = if ($WakeToRun) { " Also NDX-Trend-Paper-Wake (once a week until {0:yyyy-MM-dd}; re-run this script before then)." -f $lastWake } else { "" }
Write-Host "Installed NDX-Trend-Paper-Tick (every $EveryMinutes min, at log-on and on wake) and NDX-Trend-Paper-Watchdog (hourly).$extra"
Write-Host "Runner: $pythonw `"$paper\runner.py`" tick   (working folder $root, data $dataDir)"
Write-Host "Friday's planned trade: window 15:40 New York time = $windowText."
Write-Host "If the PC is off or asleep then, the decision is made as soon as it is on again and trades at the next close."

# ---- read-only power check (information only: catch-up covers a sleeping PC)
function Get-PowerIndex([string]$Setting) {
    $txt = (powercfg /query SCHEME_CURRENT SUB_SLEEP $Setting) -join "`n"
    $hex = [regex]::Matches($txt, "0x[0-9a-fA-F]{8}") | ForEach-Object { [Convert]::ToInt32($_.Value, 16) }
    if ($hex.Count -lt 2) { throw "unexpected powercfg output" }
    return @{ AC = $hex[$hex.Count - 2]; DC = $hex[$hex.Count - 1] }
}
try {
    $sleep = Get-PowerIndex "STANDBYIDLE"
    $wake = Get-PowerIndex "RTCWAKE"
    $wakeName = @{ 0 = "Disable"; 1 = "Enable"; 2 = "Important wake timers only" }
    if ($sleep.AC -ne 0 -or $sleep.DC -ne 0) {
        Write-Host ("Note: this PC sleeps after {0} min idle on mains power ({1} min on battery). That is fine - a decision missed while it sleeps is made when it wakes. For the exact Friday-close trade, keep it awake at {2}, or re-run with -WakeToRun and set 'Allow wake timers' to Enable." -f [math]::Round($sleep.AC / 60), [math]::Round($sleep.DC / 60), $windowText)
    }
    if ($WakeToRun -and $wake.AC -ne 1) {
        Write-Warning ("-WakeToRun has no effect until 'Allow wake timers' is set to Enable (now: {0} on mains power). Power Options > Change advanced power settings > Sleep > Allow wake timers." -f $wakeName[$wake.AC])
    }
} catch {
    Write-Host "Could not read the power settings ($($_.Exception.Message)); a missed decision is caught up when the PC is next on."
}
Write-Host "Check any time with:  python paper\runner.py status"
