"""
Alerts without any account or credentials:
  * data/alerts.log           every alert, timestamped (append-only)
  * data/ALERT_ACTIVE.txt     stays until you run `python paper/runner.py ack` (status shows it)
  * a Windows desktop notification (best effort; set alerts.desktop_notifications = false to disable)
Optional and OFF unless you add your own URLs to config.toml: a webhook (POST JSON {"text": ...}, the
Slack format; Discord expects {"content": ...}) and a dead-man "healthcheck" ping sent after each
successful weekly decision. Delivery failures are written to alerts.log (never the URL itself).
Alerting never raises: a failure to alert must not stop the run.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re

# subprocess and urllib are imported where they are used: an idle scheduled run should not pay for them

PS_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
NO_TOAST_ENV = "NDX_PAPER_NO_TOAST"     # set (e.g. by the tests) to suppress real notifications


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


class Alerter:
    def __init__(self, cfg: dict, data_dir: str, quiet: bool = False):
        self.cfg, self.data_dir, self.quiet = cfg, data_dir, quiet
        self.log_path = os.path.join(data_dir, "alerts.log")
        self.active_path = os.path.join(data_dir, "ALERT_ACTIVE.txt")

    def _append(self, path, line) -> bool:
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except OSError:
            return False

    def alert(self, subject: str, body: str = "", level: str = "WARN", notify: bool = True) -> dict:
        """Log the alert; WARN/ERROR also go to ALERT_ACTIVE.txt. notify=False keeps it out of the desktop
        notification and the webhook (for routine information such as a decision made late)."""
        line = f"{_stamp()} [{level}] {subject}" + (f" | {body}" if body else "")
        res = {"alerts.log": self._append(self.log_path, line)}
        if level in ("WARN", "ERROR"):
            res["ALERT_ACTIVE.txt"] = self._append(self.active_path, line)
        if self.quiet or not notify:
            return res
        a = self.cfg.get("alerts", {})
        if a.get("desktop_notifications") is True and os.name == "nt":
            res["desktop"] = self._toast(f"NDX trend paper: {subject}", body[:250])
        if a.get("webhook_url"):
            ok, info = self._post(a["webhook_url"], {"text": local_paths_removed(line, self.data_dir)})
            res["webhook"] = ok
            if not ok:
                self._append(self.log_path, f"{_stamp()} [INFO] webhook delivery failed: {info}")
        return res

    def ping(self) -> bool | None:
        url = self.cfg.get("alerts", {}).get("healthcheck_url")
        if not url or self.quiet:
            return None
        import urllib.request
        try:
            urllib.request.urlopen(url, timeout=10).read()
            return True
        except Exception as e:
            self._append(self.log_path, f"{_stamp()} [INFO] healthcheck ping failed: {_why(e)}")
            return False

    def acknowledge(self) -> bool:
        if os.path.exists(self.active_path):
            os.replace(self.active_path, self.active_path.replace(".txt", f"_acked_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"))
            return True
        return False

    def active(self) -> list:
        if not os.path.exists(self.active_path):
            return []
        with open(self.active_path, encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f if l.strip()]

    def withdraw(self, needle: str, reason: str) -> int:
        """Remove active alerts containing `needle` (e.g. 'missed' alerts for a day that turned out to be a
        market closure) and log why. Returns the number withdrawn."""
        try:
            lines = self.active()
            keep = [l for l in lines if needle not in l]
            n = len(lines) - len(keep)
            if n:
                tmp = self.active_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(l + "\n" for l in keep)
                os.replace(tmp, self.active_path)
                if not keep:
                    os.remove(self.active_path)
                self._append(self.log_path, f"{_stamp()} [INFO] withdrew {n} active alert(s) about '{needle}' | {reason}")
            return n
        except OSError:
            return 0

    # ------------------------------------------------------------------ channels
    @staticmethod
    def _toast(title: str, body: str) -> bool:
        """Windows toast via the built-in WinRT API (no modules). Never raises."""
        if os.environ.get(NO_TOAST_ENV):
            return False
        script = (
            "$ErrorActionPreference='Stop';"
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
            "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
            "$x = $t.GetElementsByTagName('text');"
            "$x.Item(0).AppendChild($t.CreateTextNode($env:NDX_TOAST_TITLE)) > $null;"
            "$x.Item(1).AppendChild($t.CreateTextNode($env:NDX_TOAST_BODY)) > $null;"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:NDX_TOAST_APP).Show("
            "[Windows.UI.Notifications.ToastNotification]::new($t))"
        )
        env = dict(os.environ, NDX_TOAST_TITLE=title, NDX_TOAST_BODY=body, NDX_TOAST_APP=PS_APP_ID)
        import subprocess
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                               env=env, timeout=20, capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return r.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _post(url: str, payload: dict):
        import urllib.request
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            return True, "ok"
        except Exception as e:
            return False, _why(e)


_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_paths_removed(text: str, data_dir: str | None = None) -> str:
    """The text with local folders replaced by short labels, for anything sent off this PC (the webhook): local
    paths contain the Windows user name. The local logs keep the full paths."""
    folders = [(data_dir, "<data>"), (_PROJECT, "<project>"), (os.path.expanduser("~"), "~")]
    for folder, label in folders:
        parts = [p for p in re.split(r"[\\/]+", folder or "") if p]
        if len(parts) < 2:
            continue
        pattern = r"[\\/]+".join(re.escape(p) for p in parts)     # any separators: \, \\ (escaped) or /
        text = re.sub(pattern, lambda m: label, text, flags=re.IGNORECASE)
    return text


def _why(e: Exception) -> str:
    """Failure reason without the URL (which may contain a secret token)."""
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code} {e.reason}"
    if isinstance(e, urllib.error.URLError):
        return f"network error: {e.reason}"
    return type(e).__name__
