import os

import matplotlib.pyplot as plt
import streamlit as st

from src.analytics import (
    build_crossing_dataframe,
    build_interval_dataframe,
    calculate_traffic_summary,
    create_cumulative_line_chart,
    create_direction_bar_chart,
    create_interval_bar_chart,
    summary_to_dataframe,
)
from src.motion_detector import create_motion_preview
from src.object_detector import detect_moving_objects
from src.tracking_processor import create_tracking_preview
from src.video_processor import (
    get_video_metadata,
    read_video_frame,
    save_uploaded_video,
)
from src.video_tracker import process_tracking_video


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="TrafficFlowCV",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------
# SESSION-STATE KEYS
# ---------------------------------------------------------
RESULT_KEYS = [
    "analysis_signature",
    "analysis_result",
    "analysis_video_bytes",
    "analysis_video_filename",
]


def clear_analysis_result():
    """
    Remove the previously stored analysis result.
    """

    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
with st.sidebar:

    st.header("Analysis Settings")

    st.caption(
        "Configure how the complete traffic video "
        "will be processed."
    )

    minimum_area = st.slider(
        label="Minimum moving-region area",
        min_value=100,
        max_value=3000,
        value=500,
        step=100,
        help=(
            "Small detected regions below this pixel area "
            "are ignored. Increase it to reduce noise."
        ),
    )

    roi_top_percent = st.slider(
        label="Road ROI starts at (% from top)",
        min_value=10,
        max_value=50,
        value=35,
        step=5,
        help=(
            "ROI means Region of Interest. Only the road "
            "region below this position will be analyzed."
        ),
    )

    minimum_line_percent = min(
        roi_top_percent + 15,
        80,
    )

    default_line_percent = max(
        minimum_line_percent,
        70,
    )

    default_line_percent = min(
        default_line_percent,
        90,
    )

    counting_line_percent = st.slider(
        label="Counting line position (% from top)",
        min_value=minimum_line_percent,
        max_value=90,
        value=default_line_percent,
        step=5,
        help=(
            "A tracked object is counted when its centroid "
            "crosses this horizontal line."
        ),
    )

    background_learning_frames = st.slider(
        label="Background-learning frames",
        min_value=30,
        max_value=150,
        value=90,
        step=30,
        help=(
            "Initial frames are used to learn the stationary "
            "background. They are not used for counting."
        ),
    )

    analytics_interval_seconds = st.select_slider(
        label="Analytics time interval",
        options=[1, 2, 5, 10, 15, 30],
        value=5,
        format_func=lambda value: f"{value} seconds",
        help=(
            "Crossing events will be grouped into intervals "
            "of this duration for the traffic charts."
        ),
    )

    st.divider()

    st.info(
        "Recommended initial settings:\n\n"
        "• Minimum area: 500\n\n"
        "• ROI start: 35%\n\n"
        "• Counting line: 70%\n\n"
        "• Background learning: 90 frames"
    )

    if st.button(
        "Clear Current Result",
        use_container_width=True,
    ):
        clear_analysis_result()
        st.success("Stored result cleared.")


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------
st.title("TrafficFlowCV")

st.subheader(
    "CPU-Based Traffic Video Analytics"
)

st.write(
    "Upload a fixed-camera traffic video to detect moving "
    "regions, track objects, estimate line crossings, and "
    "generate downloadable traffic analytics."
)


# ---------------------------------------------------------
# VIDEO UPLOADER
# ---------------------------------------------------------
uploaded_video = st.file_uploader(
    label="Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"],
)


# ---------------------------------------------------------
# INITIAL PAGE
# ---------------------------------------------------------
if uploaded_video is None:

    st.info(
        "Upload an MP4, AVI, MOV, or MKV traffic video "
        "to begin."
    )

    introduction_column1, introduction_column2 = (
        st.columns(2)
    )

    with introduction_column1:

        st.subheader("Application Features")

        st.write(
            """
            - Video metadata extraction
            - OpenCV motion detection
            - Centroid-based object tracking
            - Road Region of Interest filtering
            - Counting-line crossing estimation
            - Pandas traffic records
            - Matplotlib traffic charts
            - CSV and processed-video downloads
            """
        )

    with introduction_column2:

        st.subheader("Best Video Conditions")

        st.write(
            """
            - Fixed camera
            - Clearly visible roadway
            - Moderate traffic density
            - Limited vehicle overlap
            - Stable lighting
            - Vehicles crossing a visible road section
            """
        )

    st.stop()


# ---------------------------------------------------------
# SAVE VIDEO AND READ METADATA
# ---------------------------------------------------------
try:
    video_path = save_uploaded_video(
        uploaded_video
    )

    metadata = get_video_metadata(
        video_path
    )

except Exception as error:
    st.error(
        "The uploaded video could not be opened."
    )
    st.exception(error)
    st.stop()


total_frames = metadata["total_frames"]
fps = metadata["fps"]
frame_width = metadata["frame_width"]
frame_height = metadata["frame_height"]
duration_seconds = metadata[
    "duration_seconds"
]

file_size_mb = (
    uploaded_video.size / (1024 * 1024)
)


# ---------------------------------------------------------
# CURRENT PROCESSING SIGNATURE
# ---------------------------------------------------------
current_analysis_signature = (
    f"{uploaded_video.name}:"
    f"{uploaded_video.size}:"
    f"{minimum_area}:"
    f"{roi_top_percent}:"
    f"{counting_line_percent}:"
    f"{background_learning_frames}"
)


# ---------------------------------------------------------
# APPLICATION TABS
# ---------------------------------------------------------
upload_tab, results_tab, technical_tab = st.tabs(
    [
        "Upload & Analyze",
        "Results Dashboard",
        "Technical Preview",
    ]
)


# =========================================================
# TAB 1: UPLOAD AND ANALYZE
# =========================================================
with upload_tab:

    st.subheader("Uploaded Video")

    preview_column, details_column = (
        st.columns([1.5, 1])
    )

    with preview_column:

        st.video(
            uploaded_video,
            width=700,
        )

    with details_column:

        st.success(
            "Video uploaded successfully."
        )

        st.write(
            "**File name:**",
            uploaded_video.name,
        )

        st.write(
            "**File size:**",
            f"{file_size_mb:.2f} MB",
        )

        metadata_column1, metadata_column2 = (
            st.columns(2)
        )

        metadata_column1.metric(
            label="Frames",
            value=f"{total_frames:,}",
        )

        metadata_column2.metric(
            label="FPS",
            value=f"{fps:.2f}",
        )

        metadata_column3, metadata_column4 = (
            st.columns(2)
        )

        metadata_column3.metric(
            label="Resolution",
            value=(
                f"{frame_width} × "
                f"{frame_height}"
            ),
        )

        metadata_column4.metric(
            label="Duration",
            value=f"{duration_seconds:.2f} s",
        )

    st.divider()

    st.subheader("Selected Analysis Configuration")

    (
        setting_column1,
        setting_column2,
        setting_column3,
        setting_column4,
    ) = st.columns(4)

    setting_column1.metric(
        label="Minimum Area",
        value=f"{minimum_area} px²",
    )

    setting_column2.metric(
        label="ROI Start",
        value=f"{roi_top_percent}%",
    )

    setting_column3.metric(
        label="Counting Line",
        value=f"{counting_line_percent}%",
    )

    setting_column4.metric(
        label="Background Frames",
        value=background_learning_frames,
    )

    st.caption(
        "You can change these values from the sidebar "
        "before running the analysis."
    )

    st.divider()

    st.subheader("Run Traffic Analysis")

    st.write(
        "The application will process every frame, create "
        "a road ROI, track moving regions, detect counting-line "
        "crossings, and generate an annotated output video."
    )

    analyze_button = st.button(
        "Analyze Traffic Video",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        progress_bar = st.progress(0)
        progress_message = st.empty()

        def update_analysis_progress(
            progress_fraction,
            current_frame,
            video_frame_count,
        ):
            """
            Update the full-video processing progress.
            """

            progress_percentage = min(
                int(progress_fraction * 100),
                100,
            )

            progress_bar.progress(
                progress_percentage
            )

            progress_message.write(
                f"Processing frame {current_frame} "
                f"of {video_frame_count} "
                f"({progress_percentage}%)"
            )

        try:
            with st.spinner(
                "Analyzing the complete traffic video..."
            ):
                analysis_result = (
                    process_tracking_video(
                        video_path=video_path,
                        minimum_area=minimum_area,
                        warmup_frames=(
                            background_learning_frames
                        ),
                        resize_width=640,
                        roi_top_ratio=(
                            roi_top_percent / 100
                        ),
                        counting_line_ratio=(
                            counting_line_percent / 100
                        ),
                        progress_callback=(
                            update_analysis_progress
                        ),
                    )
                )

            output_video_path = (
                analysis_result["output_path"]
            )

            if not os.path.isfile(
                output_video_path
            ):
                raise FileNotFoundError(
                    "Processing finished, but the "
                    "output video was not created."
                )

            if os.path.getsize(
                output_video_path
            ) <= 0:
                raise ValueError(
                    "The output video was created, "
                    "but it is empty."
                )

            with open(
                output_video_path,
                "rb",
            ) as processed_video_file:
                processed_video_bytes = (
                    processed_video_file.read()
                )

            st.session_state[
                "analysis_signature"
            ] = current_analysis_signature

            st.session_state[
                "analysis_result"
            ] = analysis_result

            st.session_state[
                "analysis_video_bytes"
            ] = processed_video_bytes

            st.session_state[
                "analysis_video_filename"
            ] = os.path.basename(
                output_video_path
            )

            progress_bar.progress(100)

            progress_message.success(
                "Traffic analysis completed."
            )

            st.success(
                "Open the Results Dashboard tab "
                "to inspect and download the results."
            )

        except Exception as error:
            st.error(
                "The traffic-video analysis failed."
            )
            st.exception(error)


# =========================================================
# TAB 2: RESULTS DASHBOARD
# =========================================================
with results_tab:

    stored_signature = (
        st.session_state.get(
            "analysis_signature"
        )
    )

    stored_result = (
        st.session_state.get(
            "analysis_result"
        )
    )

    stored_video_bytes = (
        st.session_state.get(
            "analysis_video_bytes"
        )
    )

    stored_video_filename = (
        st.session_state.get(
            "analysis_video_filename"
        )
    )

    result_is_available = (
        stored_signature
        == current_analysis_signature
        and stored_result is not None
        and stored_video_bytes is not None
        and stored_video_filename is not None
    )

    if not result_is_available:

        st.info(
            "No result is available for the current video "
            "and settings. Open the Upload & Analyze tab "
            "and click Analyze Traffic Video."
        )

    else:

        st.subheader("Traffic Summary")

        (
            result_metric1,
            result_metric2,
            result_metric3,
            result_metric4,
        ) = st.columns(4)

        result_metric1.metric(
            label="Estimated Crossings",
            value=stored_result[
                "total_count"
            ],
        )

        result_metric2.metric(
            label="Downward",
            value=stored_result[
                "downward_count"
            ],
        )

        result_metric3.metric(
            label="Upward",
            value=stored_result[
                "upward_count"
            ],
        )

        result_metric4.metric(
            label="Frames Processed",
            value=stored_result[
                "processed_frames"
            ],
        )

        st.divider()

        st.subheader("Processed Traffic Video")

        st.video(
            stored_video_bytes,
            format="video/mp4",
            width=700,
        )

        (
            output_column1,
            output_column2,
            output_column3,
        ) = st.columns(3)

        output_column1.metric(
            label="Tracking IDs",
            value=stored_result[
                "total_registered_ids"
            ],
        )

        output_column2.metric(
            label="Maximum Visible Objects",
            value=stored_result[
                "maximum_visible_objects"
            ],
        )

        output_column3.metric(
            label="Output Size",
            value=(
                f"{len(stored_video_bytes) / (1024 * 1024):.2f} MB"
            ),
        )

        st.download_button(
            label="Download Processed Video",
            data=stored_video_bytes,
            file_name=stored_video_filename,
            mime="video/mp4",
            type="primary",
            key="results_video_download",
        )

        # -------------------------------------------------
        # CROSSING EVENT DATA
        # -------------------------------------------------
        crossing_dataframe = (
            build_crossing_dataframe(
                stored_result[
                    "crossing_events"
                ]
            )
        )

        st.divider()

        st.subheader("Crossing Event Records")

        if crossing_dataframe.empty:

            st.warning(
                "No valid counting-line crossing was recorded. "
                "Try changing the ROI, counting line, minimum "
                "area, or background-learning settings."
            )

        else:

            st.dataframe(
                crossing_dataframe,
                use_container_width=True,
                hide_index=True,
            )

            crossing_csv = (
                crossing_dataframe.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label="Download Crossing Records CSV",
                data=crossing_csv,
                file_name=(
                    "traffic_crossing_records.csv"
                ),
                mime="text/csv",
                key="results_crossing_csv",
            )

            # ---------------------------------------------
            # ANALYTICS
            # ---------------------------------------------
            interval_dataframe = (
                build_interval_dataframe(
                    crossing_dataframe=(
                        crossing_dataframe
                    ),
                    duration_seconds=(
                        stored_result[
                            "duration_seconds"
                        ]
                    ),
                    interval_seconds=(
                        analytics_interval_seconds
                    ),
                )
            )

            traffic_summary = (
                calculate_traffic_summary(
                    crossing_dataframe=(
                        crossing_dataframe
                    ),
                    interval_dataframe=(
                        interval_dataframe
                    ),
                    duration_seconds=(
                        stored_result[
                            "duration_seconds"
                        ]
                    ),
                )
            )

            st.divider()

            st.subheader("Traffic Analytics")

            (
                analytics_metric1,
                analytics_metric2,
                analytics_metric3,
                analytics_metric4,
            ) = st.columns(4)

            analytics_metric1.metric(
                label="Total Events",
                value=traffic_summary[
                    "Total Crossings"
                ],
            )

            analytics_metric2.metric(
                label="Average per Minute",
                value=traffic_summary[
                    "Average Crossings per Minute"
                ],
            )

            analytics_metric3.metric(
                label="Peak Interval",
                value=traffic_summary[
                    "Peak Interval"
                ],
            )

            analytics_metric4.metric(
                label="Peak Count",
                value=traffic_summary[
                    "Peak Interval Crossings"
                ],
            )

            st.subheader(
                "Counts by Time Interval"
            )

            st.dataframe(
                interval_dataframe,
                use_container_width=True,
                hide_index=True,
            )

            (
                interval_chart_tab,
                cumulative_chart_tab,
                direction_chart_tab,
            ) = st.tabs(
                [
                    "Crossings by Interval",
                    "Cumulative Count",
                    "Direction Comparison",
                ]
            )

            with interval_chart_tab:

                interval_figure = (
                    create_interval_bar_chart(
                        interval_dataframe
                    )
                )

                st.pyplot(
                    interval_figure,
                    use_container_width=True,
                )

                plt.close(
                    interval_figure
                )

            with cumulative_chart_tab:

                cumulative_figure = (
                    create_cumulative_line_chart(
                        crossing_dataframe
                    )
                )

                st.pyplot(
                    cumulative_figure,
                    use_container_width=True,
                )

                plt.close(
                    cumulative_figure
                )

            with direction_chart_tab:

                direction_figure = (
                    create_direction_bar_chart(
                        crossing_dataframe
                    )
                )

                st.pyplot(
                    direction_figure,
                    use_container_width=True,
                )

                plt.close(
                    direction_figure
                )

            # ---------------------------------------------
            # ANALYTICS DOWNLOADS
            # ---------------------------------------------
            summary_dataframe = (
                summary_to_dataframe(
                    traffic_summary
                )
            )

            summary_csv = (
                summary_dataframe.to_csv(
                    index=False
                ).encode("utf-8")
            )

            interval_csv = (
                interval_dataframe.to_csv(
                    index=False
                ).encode("utf-8")
            )

            download_column1, download_column2 = (
                st.columns(2)
            )

            with download_column1:

                st.download_button(
                    label="Download Traffic Summary CSV",
                    data=summary_csv,
                    file_name="traffic_summary.csv",
                    mime="text/csv",
                    key="results_summary_csv",
                    use_container_width=True,
                )

            with download_column2:

                st.download_button(
                    label="Download Interval Counts CSV",
                    data=interval_csv,
                    file_name=(
                        "traffic_interval_counts.csv"
                    ),
                    mime="text/csv",
                    key="results_interval_csv",
                    use_container_width=True,
                )

        st.warning(
            "The results are estimates produced by classical "
            "motion detection and centroid tracking. They "
            "should not be treated as verified ground-truth "
            "vehicle counts."
        )


# =========================================================
# TAB 3: TECHNICAL PREVIEW
# =========================================================
with technical_tab:

    st.subheader("Technical Processing Preview")

    st.info(
        "This section is optional. It shows how the "
        "computer-vision pipeline works internally."
    )

    maximum_frame_number = max(
        total_frames - 1,
        0,
    )

    default_frame_number = min(
        total_frames // 2,
        maximum_frame_number,
    )

    selected_frame_number = st.slider(
        label="Select a frame for technical inspection",
        min_value=0,
        max_value=maximum_frame_number,
        value=default_frame_number,
        step=1,
        key="technical_frame_slider",
    )

    selected_frame_rgb = read_video_frame(
        video_path,
        selected_frame_number,
    )

    if selected_frame_rgb is not None:

        selected_time_seconds = (
            selected_frame_number / fps
            if fps > 0
            else 0
        )

        st.image(
            selected_frame_rgb,
            caption=(
                f"Frame {selected_frame_number} "
                f"at {selected_time_seconds:.2f} seconds"
            ),
            width=700,
        )

        st.caption(
            f"NumPy frame-array shape: "
            f"{selected_frame_rgb.shape}"
        )

    st.divider()

    technical_detection_button = st.button(
        "Generate Motion and Detection Preview",
        key="technical_detection_button",
    )

    if technical_detection_button:

        try:
            with st.spinner(
                "Generating the motion mask and "
                "candidate-object preview..."
            ):
                (
                    motion_frame_rgb,
                    motion_mask,
                    processed_frame_count,
                ) = create_motion_preview(
                    video_path=video_path,
                    target_frame_number=(
                        selected_frame_number
                    ),
                )

                (
                    annotated_frame_rgb,
                    detections,
                ) = detect_moving_objects(
                    frame_rgb=motion_frame_rgb,
                    motion_mask=motion_mask,
                    minimum_area=minimum_area,
                )

            motion_column, mask_column = (
                st.columns(2)
            )

            with motion_column:

                st.image(
                    motion_frame_rgb,
                    caption="Traffic frame",
                    use_container_width=True,
                )

            with mask_column:

                st.image(
                    motion_mask,
                    caption="Cleaned motion mask",
                    use_container_width=True,
                )

            st.image(
                annotated_frame_rgb,
                caption=(
                    "Candidate moving-region "
                    "bounding boxes"
                ),
                width=700,
            )

            detection_metric1, detection_metric2 = (
                st.columns(2)
            )

            detection_metric1.metric(
                label="Candidate Regions",
                value=len(detections),
            )

            detection_metric2.metric(
                label="Frames Used",
                value=processed_frame_count,
            )

        except Exception as error:
            st.error(
                "The detection preview failed."
            )
            st.exception(error)

    st.divider()

    tracking_frames = st.slider(
        label="Tracking-preview frames",
        min_value=30,
        max_value=180,
        value=90,
        step=30,
        key="technical_tracking_frames",
    )

    technical_tracking_button = st.button(
        "Generate Tracking Preview",
        key="technical_tracking_button",
    )

    if technical_tracking_button:

        try:
            with st.spinner(
                "Processing consecutive frames and "
                "assigning tracking IDs..."
            ):
                tracking_result = (
                    create_tracking_preview(
                        video_path=video_path,
                        target_frame_number=(
                            selected_frame_number
                        ),
                        minimum_area=minimum_area,
                        tracking_frames=(
                            tracking_frames
                        ),
                    )
                )

            st.image(
                tracking_result[
                    "annotated_frame"
                ],
                caption=(
                    "Centroid-tracking preview at "
                    f"frame {selected_frame_number}"
                ),
                width=700,
            )

            (
                tracking_metric1,
                tracking_metric2,
                tracking_metric3,
            ) = st.columns(3)

            tracking_metric1.metric(
                label="Objects Visible",
                value=tracking_result[
                    "visible_objects"
                ],
            )

            tracking_metric2.metric(
                label="IDs Registered",
                value=tracking_result[
                    "registered_ids"
                ],
            )

            tracking_metric3.metric(
                label="Frames Processed",
                value=tracking_result[
                    "processed_frames"
                ],
            )

        except Exception as error:
            st.error(
                "The tracking preview failed."
            )
            st.exception(error)