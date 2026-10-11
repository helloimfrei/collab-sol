from pathlib import Path

import pandas as pd

from collab_sol.data import build_incident_targets, prepare_crossing_year


def test_incident_targets_use_direct_form57_casualties(tmp_path: Path) -> None:
    form57 = pd.DataFrame(
        {
            "gradecrossingid": ["A", "B", "C"],
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "totalkilledform57": [1, 0, 0],
            "totalinjuredform57": [0, 2, 0],
            "totalkilledform55a": [0, 1, 1],
            "totalinjuredform55a": [0, 0, 1],
        }
    )
    source = tmp_path / "form57.csv"
    form57.to_csv(source, index=False)

    targets = build_incident_targets(source).set_index("crossingid")

    assert targets.loc["A", "severity_target"] == "fatal"
    assert targets.loc["B", "severity_target"] == "injury"
    assert targets.loc["C", "severity_target"] == "property_damage_only"


def test_prepare_crossing_year_deduplicates_before_filtering() -> None:
    frame = pd.DataFrame(
        {
            "crossingid": ["A", "A", "B", "C"],
            "asof_year": [2024, 2024, 2024, 2024],
            "revisiondate": ["2020-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
            "latitude": [40.0, 40.0, 41.0, 42.0],
            "longitude": [-75.0, -75.0, -76.0, -77.0],
            "crossingtypecode": [3, 3, 3, 2],
            "crossingpositioncode": [1, 1, 1, 1],
            "crossingclosed": ["No", "Yes", "No", "No"],
            "incident": [0, 0, 0, 0],
            "incident_count": [0, 0, 0, 0],
        }
    )
    targets = pd.DataFrame(
        {
            "crossingid": ["B"],
            "target_year": [2024],
            "incident_count": [1],
            "incident": [1],
            "total_killed": [0],
            "total_injured": [2],
            "severity_ordinal": [2],
            "severity_target": ["injury"],
        }
    )

    result = prepare_crossing_year(frame, targets)

    assert result["crossingid"].tolist() == ["B"]
    assert result.loc[0, "incident"] == 1
    assert result.loc[0, "severity_target"] == "injury"
