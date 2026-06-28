"""Zone-lookup reference loader — Phase 1 acceptance criterion (3), offline."""

from __future__ import annotations

import pytest

from fleetcast.data.reference import ZONE_LOOKUP_COLUMNS, load_zone_lookup, zone_names


def test_zone_lookup_loads_offline(make_synthetic_env):
    env = make_synthetic_env()  # fixture writes a synthetic taxi_zone_lookup.csv
    df = load_zone_lookup(env.cfg)
    assert list(df.columns)[: len(ZONE_LOOKUP_COLUMNS)] == ZONE_LOOKUP_COLUMNS
    assert df["LocationID"].dtype.kind in "iu"
    assert {1, 100, 263} <= set(df["LocationID"])  # 1..263 covered


def test_zone_names_mapping(make_synthetic_env):
    env = make_synthetic_env()
    names = zone_names(env.cfg)
    assert names[1].startswith("Manhattan — ")


def test_missing_lookup_raises(make_synthetic_env):
    env = make_synthetic_env()
    (env.raw_dir / "taxi_zone_lookup.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_zone_lookup(env.cfg)
