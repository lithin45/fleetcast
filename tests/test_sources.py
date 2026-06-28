"""URL builders and month/date helpers."""

from __future__ import annotations

from datetime import date

from fleetcast.data import sources


def test_trip_data_url_and_filename():
    url = sources.trip_data_url("yellow", "2024-01")
    assert url.endswith("/trip-data/yellow_tripdata_2024-01.parquet")
    assert sources.trip_data_filename("green", "2024-03") == "green_tripdata_2024-03.parquet"


def test_month_bounds_handles_leap_year():
    first, last = sources.month_bounds("2024-02")
    assert first == date(2024, 2, 1)
    assert last == date(2024, 2, 29)  # 2024 is a leap year


def test_months_date_range_spans_all():
    start, end = sources.months_date_range(["2024-03", "2024-01", "2024-02"])
    assert start == date(2024, 1, 1)
    assert end == date(2024, 3, 31)
