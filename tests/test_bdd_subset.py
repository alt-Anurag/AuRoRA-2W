"""Subset membership only; no model predictions or optimization."""

from scripts.prepare_bdd_subset import select_groups


def test_selection_is_deterministic_and_retains_whole_groups():
    rows = [{"id": str(i), "sequence": str(i // 2), "attributes": {"weather": "clear", "timeofday": "day"}} for i in range(20)]
    selected = select_groups(rows, 7, 42)
    assert selected == select_groups(list(reversed(rows)), 7, 42)
    assert len(selected) == 8  # Whole group crossing the quota stays together.
    for group in {row["sequence"] for row in selected}:
        assert sum(row["sequence"] == group for row in selected) == 2


def test_singleton_stratified_subset_matches_exact_budget_and_proportions():
    rows = [{"id": str(i), "sequence": str(i), "attributes": {"timeofday": "night" if i < 20 else "day"}} for i in range(100)]
    selected = select_groups(rows, 10, 7)
    assert len(selected) == 10
    assert sum(row["attributes"]["timeofday"] == "night" for row in selected) == 2


def test_archive_strata_do_not_invent_weather_and_preserve_both_parts():
    rows = [{"id": str(i), "sequence": str(i), "source_archive_name": "part1" if i < 40 else "part2"} for i in range(100)]
    selected = select_groups(rows, 10, 42, stratum_key="source_archive_name")
    assert len(selected) == 10
    assert sum(row["source_archive_name"] == "part1" for row in selected) == 4
    assert all("attributes" not in row for row in selected)
