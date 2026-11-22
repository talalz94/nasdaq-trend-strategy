import math
import os

import pytest

from paper.config import load_config


def cfg_with(tmp_path, text):
    p = os.path.join(str(tmp_path), "c.toml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return load_config(p)


def test_defaults_are_valid(tmp_path):
    cfg = load_config(os.path.join(str(tmp_path), "absent.toml"))
    assert cfg["strategy"]["tier"] == "F2" and cfg["schedule"]["window_minutes"] == 8


@pytest.mark.parametrize("text", [
    "[strategy]\ntier = \"f2\"",                                  # wrong case
    "[strategy]\nteir = \"F1\"",                                  # typo in a key
    "[strategy]\ntier = \"F3\"\nroute = \"UCITS\"",
    "[strategy]\nband = 0.5",
    "[strategy]\nband = \"0.02\"",
    "[account]\ncost_bps = nan",
    "[account]\ncommission_per_order = inf",
    "[account]\nstarting_cash = nan",
    "[account]\nfractional_shares = \"false\"",
    "[alerts]\ndesktop_notifications = \"false\"",
    "[schedule]\nsignal_minutes_before_close = 0",                # window after the close: look-ahead
    "[schedule]\nsignal_minutes_before_close = 15",               # window ends after the MOC cut-off
    "[schedule]\nwindow_minutes = 0",
    "[schedule]\nsettle_minutes_after_close = -60",
    "[schedule]\nextra_holidays = [\"2025-1-9\"]",
    "[schedule]\nextra_holidays = \"2025-01-09\"",
    "[review]\ndrawdown_trigger = { F2 = -0.35 }",
    "[review]\ndrawdown_trigger = 0.35",
    "[review]\nrolling_12m_flag = 25",
    "[alerts]\nwebhook_url = \"hooks.example.com/x\"",
    "tier = \"F2\"",                                              # key outside any section
    "[strategy\ntier = \"F2\"",                                   # TOML syntax error
])
def test_bad_settings_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        cfg_with(tmp_path, text)


def test_good_settings_are_accepted(tmp_path):
    cfg = cfg_with(tmp_path, "[strategy]\ntier = \"F1\"\n[schedule]\nextra_holidays = [\"2025-01-09\", 2018-12-05]\n"
                             "[review]\ndrawdown_trigger = { F1 = 0.2 }\n[alerts]\nwebhook_url = \"https://example.com/h\"")
    assert cfg["strategy"]["tier"] == "F1" and len(cfg["schedule"]["extra_holidays"]) == 2


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(os.path.join(str(tmp_path), "nope.toml"), must_exist=True)
