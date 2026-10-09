from src.data_loader import DataLoader


def test_load_returns_real_event_count():
    loader = DataLoader()
    events = loader.load()
    assert len(events) == 104


def test_load_covers_four_months():
    loader = DataLoader()
    events = loader.load()
    assert events["Date Out"].min().month == 1
    assert events["Date Out"].max().month == 4


def test_load_has_three_feeders():
    loader = DataLoader()
    events = loader.load()
    assert events["Affected Feeder"].nunique() == 3


def test_load_no_missing_critical_fields():
    loader = DataLoader()
    events = loader.load()
    for col in ("Date Out", "Duration_Minutes", "Load Loss (MW)", "Affected Feeder"):
        assert events[col].isna().sum() == 0


def test_duration_minutes_parsed_correctly():
    loader = DataLoader()
    events = loader.load()
    assert events["Duration_Minutes"].min() > 0
    assert events["Duration_Minutes"].max() <= 1440  # no event longer than 1 day
