"""Assemble the final crossing-year modeling dataset."""

from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from collab_sol.weather import add_weather

LABEL_COLUMNS = [
    "incident_count",
    "incident",
    "total_killed",
    "total_injured",
    "severity_ordinal",
    "severity_target",
]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _columns_with_type_drift(files: list[Path]) -> list[str]:
    types: dict[str, set[pa.DataType]] = {}
    for path in files:
        for field in pq.read_schema(path):
            types.setdefault(field.name, set()).add(field.type)
    return sorted(column for column, observed in types.items() if len(observed) > 1)


def build_incident_targets(form57_csv: str | Path) -> pd.DataFrame:
    """Aggregate direct Form 57 risk and severity labels by crossing and year."""

    incidents = pd.read_csv(form57_csv, dtype="string", low_memory=False)
    required = [
        "gradecrossingid",
        "date",
        "totalkilledform57",
        "totalinjuredform57",
    ]
    _require_columns(incidents, required)

    incidents["crossingid"] = incidents["gradecrossingid"].str.strip().str.upper()
    incidents["target_year"] = pd.to_datetime(
        incidents["date"], errors="coerce"
    ).dt.year
    killed = cast(
        pd.Series,
        pd.to_numeric(incidents["totalkilledform57"], errors="coerce"),
    )
    injured = cast(
        pd.Series,
        pd.to_numeric(incidents["totalinjuredform57"], errors="coerce"),
    )
    incidents["total_killed"] = killed.fillna(0)
    incidents["total_injured"] = injured.fillna(0)
    incidents = incidents.dropna(subset=["crossingid", "target_year"])

    targets = pd.DataFrame(
        incidents.groupby(["crossingid", "target_year"], as_index=False).agg(
            incident_count=("crossingid", "size"),
            total_killed=("total_killed", "sum"),
            total_injured=("total_injured", "sum"),
        )
    )
    targets["incident"] = 1
    targets["severity_ordinal"] = np.select(
        [targets["total_killed"].gt(0), targets["total_injured"].gt(0)],
        [3, 2],
        default=1,
    )
    severity = cast(pd.Series, targets["severity_ordinal"])
    targets["severity_target"] = severity.map(
        {1: "property_damage_only", 2: "injury", 3: "fatal"}
    )

    integer_types = {
        "target_year": "int16",
        "incident_count": "int16",
        "incident": "int8",
        "total_killed": "int16",
        "total_injured": "int16",
        "severity_ordinal": "int8",
    }
    targets = targets.astype(integer_types)
    targets["severity_target"] = targets["severity_target"].astype("string")
    return targets


def prepare_crossing_year(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Deduplicate, filter, and label one annual crossing snapshot."""

    required = [
        "crossingid",
        "asof_year",
        "revisiondate",
        "latitude",
        "longitude",
        "crossingtypecode",
        "crossingpositioncode",
        "crossingclosed",
    ]
    _require_columns(frame, required)
    _require_columns(targets, ["crossingid", "target_year", *LABEL_COLUMNS])

    result = frame.copy()
    result["crossingid"] = result["crossingid"].astype("string").str.strip().str.upper()
    result = result.dropna(subset=["crossingid"])
    result["_revision_timestamp"] = pd.to_datetime(
        result["revisiondate"], errors="coerce"
    )
    result = result.sort_values(
        ["_revision_timestamp", "crossingid"],
        kind="stable",
        na_position="first",
    ).drop_duplicates("crossingid", keep="last")
    result = result.drop(columns="_revision_timestamp")

    latitude = cast(
        pd.Series, pd.to_numeric(result["latitude"], errors="coerce")
    )
    longitude = cast(
        pd.Series, pd.to_numeric(result["longitude"], errors="coerce")
    )
    eligible = (
        result["crossingtypecode"].eq(3)
        & result["crossingpositioncode"].eq(1)
        & result["crossingclosed"].astype("string").str.casefold().eq("no")
        & latitude.between(24, 50)
        & longitude.between(-125, -66)
    )
    result = result.loc[eligible].copy()

    existing_labels = [column for column in LABEL_COLUMNS if column in result]
    result = result.drop(columns=existing_labels)
    result = result.merge(
        targets,
        left_on=["crossingid", "asof_year"],
        right_on=["crossingid", "target_year"],
        how="left",
        validate="one_to_one",
    ).drop(columns="target_year")

    integer_types = {
        "incident_count": "int16",
        "incident": "int8",
        "total_killed": "int16",
        "total_injured": "int16",
        "severity_ordinal": "int8",
    }
    for column, dtype in integer_types.items():
        result[column] = result[column].fillna(0).astype(dtype)
    result["severity_target"] = result["severity_target"].fillna("none").astype("string")

    if result.duplicated(["crossingid", "asof_year"]).any():
        raise ValueError("Crossing-year keys are not unique after preparation.")
    missing_severity = result["incident"].eq(1) & result["severity_target"].eq("none")
    if missing_severity.any():
        raise ValueError("At least one incident is missing its severity label.")
    return result.reset_index(drop=True)


def build_master_dataset(
    annual_files: Iterable[str | Path],
    form57_csv: str | Path,
    gridmet_dir: str | Path,
    weather_lookup: str | Path,
    output: str | Path,
) -> Path:
    """Write all prepared annual snapshots into one Parquet file."""

    files = sorted(Path(path) for path in annual_files)
    if not files:
        raise ValueError("At least one annual dataset is required.")

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)

    targets = build_incident_targets(form57_csv)
    string_columns = _columns_with_type_drift(files)
    if string_columns:
        print(f"normalizing schema-drift columns as strings: {string_columns}")
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    try:
        for path in files:
            frame = pd.read_parquet(path)
            for column in string_columns:
                frame[column] = frame[column].astype("string")
            prepared = prepare_crossing_year(frame, targets)
            prepared = add_weather(prepared, gridmet_dir, weather_lookup)
            table = pa.Table.from_pandas(prepared, preserve_index=False)

            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(partial, output_schema, compression="zstd")
            elif output_schema is not None and not table.schema.equals(
                output_schema, check_metadata=False
            ):
                table = table.cast(output_schema, safe=False)
            writer.write_table(table)

            year = int(prepared["asof_year"].iloc[0])
            print(
                f"{year}: {len(prepared):,} crossings, "
                f"{int(prepared['incident'].sum()):,} incidents, "
                f"{int(prepared['gridmet_mapped'].sum()):,} weather matches"
            )
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
        partial.replace(output)
    return output


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    args = parser.parse_args()
    data_root = args.data_root
    output = build_master_dataset(
        annual_files=(data_root / "full_dataset").glob("full_dataset_*.parquet"),
        form57_csv=data_root / "raw/form57/form57_2019_2024.csv",
        gridmet_dir=data_root / "raw/gridmet",
        weather_lookup=data_root / "interim/weather/gridmet_5y_by_cell_asof.parquet",
        output=data_root / "processed/crossing_year_master.parquet",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
