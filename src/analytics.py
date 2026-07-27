from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_COLUMNS = [
    "Object ID",
    "Crossing Frame",
    "Crossing Time (s)",
    "Direction",
    "Centroid X",
    "Centroid Y",
]


def build_crossing_dataframe(
    crossing_events: list[dict] | None,
) -> pd.DataFrame:
    """
    Convert line-crossing event records into a clean
    Pandas DataFrame suitable for display and analysis.
    """

    if not crossing_events:
        return pd.DataFrame(
            columns=DISPLAY_COLUMNS
        )

    crossing_dataframe = pd.DataFrame(
        crossing_events
    )

    crossing_dataframe = crossing_dataframe.rename(
        columns={
            "object_id": "Object ID",
            "frame_number": "Crossing Frame",
            "time_seconds": "Crossing Time (s)",
            "direction": "Direction",
            "center_x": "Centroid X",
            "center_y": "Centroid Y",
        }
    )

    # Add any missing columns to prevent KeyErrors.
    for column_name in DISPLAY_COLUMNS:
        if column_name not in crossing_dataframe.columns:
            crossing_dataframe[column_name] = np.nan

    numeric_columns = [
        "Object ID",
        "Crossing Frame",
        "Crossing Time (s)",
        "Centroid X",
        "Centroid Y",
    ]

    for column_name in numeric_columns:
        crossing_dataframe[column_name] = (
            pd.to_numeric(
                crossing_dataframe[column_name],
                errors="coerce",
            )
        )

    crossing_dataframe = (
        crossing_dataframe.dropna(
            subset=[
                "Object ID",
                "Crossing Frame",
                "Crossing Time (s)",
            ]
        )
    )

    integer_columns = [
        "Object ID",
        "Crossing Frame",
        "Centroid X",
        "Centroid Y",
    ]

    for column_name in integer_columns:
        crossing_dataframe[column_name] = (
            crossing_dataframe[column_name]
            .fillna(0)
            .astype(int)
        )

    crossing_dataframe["Crossing Time (s)"] = (
        crossing_dataframe["Crossing Time (s)"]
        .astype(float)
        .round(2)
    )

    crossing_dataframe["Direction"] = (
        crossing_dataframe["Direction"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    crossing_dataframe = (
        crossing_dataframe[
            DISPLAY_COLUMNS
        ]
        .sort_values(
            by="Crossing Time (s)"
        )
        .reset_index(drop=True)
    )

    return crossing_dataframe


def build_interval_dataframe(
    crossing_dataframe: pd.DataFrame,
    duration_seconds: float,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """
    Group crossing events into equal time intervals.

    For example, with a five-second interval:

    0–5 seconds
    5–10 seconds
    10–15 seconds
    """

    if interval_seconds <= 0:
        raise ValueError(
            "The analytics interval must be positive."
        )

    video_duration = max(
        float(duration_seconds),
        0.0,
    )

    if not crossing_dataframe.empty:
        last_crossing_time = float(
            crossing_dataframe[
                "Crossing Time (s)"
            ].max()
        )

        video_duration = max(
            video_duration,
            last_crossing_time,
        )

    number_of_intervals = max(
        1,
        math.ceil(
            video_duration / interval_seconds
        ),
    )

    interval_starts = (
        np.arange(number_of_intervals)
        * interval_seconds
    )

    interval_dataframe = pd.DataFrame(
        {
            "Interval Start (s)": interval_starts
        }
    )

    interval_dataframe[
        "Interval End (s)"
    ] = (
        interval_dataframe[
            "Interval Start (s)"
        ]
        + interval_seconds
    )

    interval_dataframe[
        "Total Crossings"
    ] = 0

    interval_dataframe[
        "Upward"
    ] = 0

    interval_dataframe[
        "Downward"
    ] = 0

    if not crossing_dataframe.empty:

        working_dataframe = (
            crossing_dataframe.copy()
        )

        working_dataframe[
            "Interval Start (s)"
        ] = (
            np.floor(
                working_dataframe[
                    "Crossing Time (s)"
                ]
                / interval_seconds
            )
            * interval_seconds
        ).astype(int)

        total_counts = (
            working_dataframe.groupby(
                "Interval Start (s)"
            )
            .size()
        )

        upward_counts = (
            working_dataframe[
                working_dataframe[
                    "Direction"
                ] == "Upward"
            ]
            .groupby("Interval Start (s)")
            .size()
        )

        downward_counts = (
            working_dataframe[
                working_dataframe[
                    "Direction"
                ] == "Downward"
            ]
            .groupby("Interval Start (s)")
            .size()
        )

        interval_dataframe[
            "Total Crossings"
        ] = (
            interval_dataframe[
                "Interval Start (s)"
            ]
            .map(total_counts)
            .fillna(0)
            .astype(int)
        )

        interval_dataframe[
            "Upward"
        ] = (
            interval_dataframe[
                "Interval Start (s)"
            ]
            .map(upward_counts)
            .fillna(0)
            .astype(int)
        )

        interval_dataframe[
            "Downward"
        ] = (
            interval_dataframe[
                "Interval Start (s)"
            ]
            .map(downward_counts)
            .fillna(0)
            .astype(int)
        )

    interval_dataframe[
        "Time Interval"
    ] = interval_dataframe.apply(
        lambda row: (
            f'{int(row["Interval Start (s)"])}'
            f'–{int(row["Interval End (s)"])} s'
        ),
        axis=1,
    )

    interval_dataframe = interval_dataframe[
        [
            "Time Interval",
            "Interval Start (s)",
            "Interval End (s)",
            "Total Crossings",
            "Upward",
            "Downward",
        ]
    ]

    return interval_dataframe


def calculate_traffic_summary(
    crossing_dataframe: pd.DataFrame,
    interval_dataframe: pd.DataFrame,
    duration_seconds: float,
) -> dict:
    """
    Calculate high-level traffic statistics.
    """

    total_crossings = len(
        crossing_dataframe
    )

    upward_crossings = int(
        (
            crossing_dataframe[
                "Direction"
            ] == "Upward"
        ).sum()
    )

    downward_crossings = int(
        (
            crossing_dataframe[
                "Direction"
            ] == "Downward"
        ).sum()
    )

    video_duration = max(
        float(duration_seconds),
        0.0,
    )

    if video_duration > 0:
        average_crossings_per_minute = (
            total_crossings
            / video_duration
            * 60
        )
    else:
        average_crossings_per_minute = 0.0

    if (
        not interval_dataframe.empty
        and interval_dataframe[
            "Total Crossings"
        ].max() > 0
    ):
        peak_index = (
            interval_dataframe[
                "Total Crossings"
            ].idxmax()
        )

        peak_row = interval_dataframe.loc[
            peak_index
        ]

        peak_interval = peak_row[
            "Time Interval"
        ]

        peak_interval_crossings = int(
            peak_row["Total Crossings"]
        )
    else:
        peak_interval = "No crossings"
        peak_interval_crossings = 0

    return {
        "Total Crossings": total_crossings,
        "Upward Crossings": upward_crossings,
        "Downward Crossings": downward_crossings,
        "Average Crossings per Minute": round(
            average_crossings_per_minute,
            2,
        ),
        "Peak Interval": peak_interval,
        "Peak Interval Crossings": (
            peak_interval_crossings
        ),
        "Video Duration (s)": round(
            video_duration,
            2,
        ),
    }


def summary_to_dataframe(
    traffic_summary: dict,
) -> pd.DataFrame:
    """
    Convert the traffic summary dictionary into a
    one-row DataFrame for CSV export.
    """

    return pd.DataFrame(
        [traffic_summary]
    )


def create_interval_bar_chart(
    interval_dataframe: pd.DataFrame,
):
    """
    Create a bar chart of crossings by time interval.
    """

    figure, axis = plt.subplots(
        figsize=(9, 4.8)
    )

    axis.bar(
        interval_dataframe[
            "Time Interval"
        ],
        interval_dataframe[
            "Total Crossings"
        ],
    )

    axis.set_title(
        "Traffic Crossings by Time Interval"
    )

    axis.set_xlabel(
        "Video Time Interval"
    )

    axis.set_ylabel(
        "Number of Crossings"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure


def create_cumulative_line_chart(
    crossing_dataframe: pd.DataFrame,
):
    """
    Create a cumulative traffic-crossing line chart.
    """

    figure, axis = plt.subplots(
        figsize=(9, 4.8)
    )

    sorted_dataframe = (
        crossing_dataframe.sort_values(
            by="Crossing Time (s)"
        )
        .reset_index(drop=True)
        .copy()
    )

    sorted_dataframe[
        "Cumulative Crossings"
    ] = np.arange(
        1,
        len(sorted_dataframe) + 1,
    )

    axis.plot(
        sorted_dataframe[
            "Crossing Time (s)"
        ],
        sorted_dataframe[
            "Cumulative Crossings"
        ],
        marker="o",
    )

    axis.set_title(
        "Cumulative Traffic Crossings"
    )

    axis.set_xlabel(
        "Video Time (seconds)"
    )

    axis.set_ylabel(
        "Cumulative Count"
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    return figure


def create_direction_bar_chart(
    crossing_dataframe: pd.DataFrame,
):
    """
    Compare upward and downward crossing totals.
    """

    direction_order = [
        "Upward",
        "Downward",
    ]

    direction_counts = (
        crossing_dataframe[
            "Direction"
        ]
        .value_counts()
        .reindex(
            direction_order,
            fill_value=0,
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 4.8)
    )

    axis.bar(
        direction_counts.index,
        direction_counts.values,
    )

    axis.set_title(
        "Traffic Crossings by Direction"
    )

    axis.set_xlabel(
        "Direction"
    )

    axis.set_ylabel(
        "Number of Crossings"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure