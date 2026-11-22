import datetime as dt

from paper.nyse_calendar import EARLY_CLOSE, REGULAR_CLOSE, NYSECalendar, rule_early_closes, rule_holidays

D = dt.date
cal = NYSECalendar()


def test_holidays_2026():
    assert rule_holidays(2026) == {D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 4, 3), D(2026, 5, 25),
                                   D(2026, 6, 19), D(2026, 7, 3), D(2026, 9, 7), D(2026, 11, 26), D(2026, 12, 25)}


def test_holidays_2027_observed_rules():
    assert rule_holidays(2027) == {D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26), D(2027, 5, 31),
                                   D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6), D(2027, 11, 25), D(2027, 12, 24)}


def test_new_year_on_saturday_is_not_observed_on_friday():
    assert cal.is_session(D(2021, 12, 31))          # 1 Jan 2022 was a Saturday
    assert D(2022, 6, 20) in rule_holidays(2022)     # Juneteenth on Sunday -> Monday
    assert D(2021, 6, 18) not in rule_holidays(2021)  # Juneteenth only from 2022


def test_early_closes():
    assert rule_early_closes(2026) == {D(2026, 11, 27), D(2026, 12, 24)}      # 3 July 2026 is the holiday
    assert rule_early_closes(2025) == {D(2025, 7, 3), D(2025, 11, 28), D(2025, 12, 24)}
    assert rule_early_closes(2027) == {D(2027, 11, 26)}                       # 24 Dec 2027 is the holiday
    assert cal.close_time(D(2026, 11, 27)) == EARLY_CLOSE
    assert cal.close_time(D(2026, 11, 20)) == REGULAR_CLOSE


def test_decision_days_holiday_weeks():
    assert cal.week_decision_day(D(2026, 4, 1)) == D(2026, 4, 2)     # Good Friday -> Thursday
    assert cal.week_decision_day(D(2026, 6, 29)) == D(2026, 7, 2)    # 3 July observed -> Thursday
    assert cal.week_decision_day(D(2026, 11, 23)) == D(2026, 11, 27)  # day after Thanksgiving (1 pm close)
    assert cal.week_decision_day(D(2026, 12, 21)) == D(2026, 12, 24)  # Christmas Friday -> Thursday (1 pm)
    assert cal.week_decision_day(D(2026, 12, 28)) == D(2026, 12, 31)  # New Year Friday -> Thursday
    assert cal.week_decision_day(D(2026, 11, 21)) == D(2026, 11, 20)  # Saturday belongs to its Mon-Fri week


def test_latest_and_next_decision_day():
    assert cal.latest_decision_day(D(2026, 11, 18)) == D(2026, 11, 13)
    assert cal.latest_decision_day(D(2026, 11, 20)) == D(2026, 11, 20)
    assert cal.latest_decision_day(D(2026, 11, 22)) == D(2026, 11, 20)
    assert cal.next_decision_day(D(2026, 11, 20)) == D(2026, 11, 27)
    assert cal.next_decision_day(D(2026, 11, 18)) == D(2026, 11, 20)


def test_extra_closures():
    c = NYSECalendar(extra_holidays=["2025-01-09"])
    assert not c.is_session(D(2025, 1, 9))
    assert c.prev_session(D(2025, 1, 10)) == D(2025, 1, 8)
