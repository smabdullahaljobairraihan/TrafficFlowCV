from pathlib import Path

import cv2
import numpy as np

from src.centroid_tracker import CentroidTracker
from src.motion_detector import remove_small_regions
from src.object_detector import detect_moving_objects
from src.vehicle_counter import LineCrossingCounter


def process_tracking_video(
    video_path,
    minimum_area=300,
    warmup_frames=60,
    resize_width=640,
    roi_top_ratio=0.20,
    counting_line_ratio=0.70,
    output_folder="outputs/tracked_videos",
    progress_callback=None
):
    """
    Process a complete traffic video.

    The function performs:
    1. Road region-of-interest filtering
    2. Background subtraction
    3. Moving-region detection
    4. Centroid tracking
    5. Counting-line crossing detection
    6. Annotated video generation
    """

    output_directory = Path(output_folder)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    input_video_path = Path(video_path)

    output_path = output_directory / (
        f"{input_video_path.stem}_counted.mp4"
    )

    video_capture = cv2.VideoCapture(
        str(input_video_path)
    )

    if not video_capture.isOpened():
        raise ValueError(
            "OpenCV could not open the input video."
        )

    video_writer = None

    try:
        # -------------------------------------------------
        # READ INPUT VIDEO PROPERTIES
        # -------------------------------------------------
        total_frames = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            video_capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        original_width = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        original_height = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if total_frames <= 0:
            raise ValueError(
                "The video contains no readable frames."
            )

        if fps <= 0:
            fps = 25.0

        if (
            original_width <= 0
            or original_height <= 0
        ):
            raise ValueError(
                "The video resolution could not be read."
            )

        # -------------------------------------------------
        # CALCULATE OUTPUT RESOLUTION
        # -------------------------------------------------
        if (
            resize_width is not None
            and original_width > resize_width
        ):
            resize_ratio = (
                resize_width / original_width
            )

            output_width = resize_width

            output_height = int(
                original_height * resize_ratio
            )

        else:
            output_width = original_width
            output_height = original_height

        # Video encoders usually require even dimensions
        if output_width % 2 != 0:
            output_width -= 1

        if output_height % 2 != 0:
            output_height -= 1

        # -------------------------------------------------
        # CREATE OUTPUT VIDEO
        # -------------------------------------------------
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (output_width, output_height)
        )

        if not video_writer.isOpened():

            video_writer.release()

            video_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (output_width, output_height)
            )

        if not video_writer.isOpened():
            raise ValueError(
                "OpenCV could not create the output video."
            )

        # -------------------------------------------------
        # CREATE ROAD REGION OF INTEREST
        # -------------------------------------------------
        roi_top_y = int(
            output_height * roi_top_ratio
        )

        roi_polygon = np.array(
            [
                [
                    int(output_width * 0.16),
                    roi_top_y
                ],
                [
                    int(output_width * 0.84),
                    roi_top_y
                ],
                [
                    output_width - 1,
                    output_height - 1
                ],
                [
                    0,
                    output_height - 1
                ]
            ],
            dtype=np.int32
        )

        roi_mask = np.zeros(
            (output_height, output_width),
            dtype=np.uint8
        )

        cv2.fillPoly(
            roi_mask,
            [roi_polygon],
            255
        )

        counting_line_y = int(
            output_height * counting_line_ratio
        )

        # -------------------------------------------------
        # BACKGROUND-SUBTRACTION COMPONENTS
        # -------------------------------------------------
        background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=max(warmup_frames, 150),
                varThreshold=25,
                detectShadows=True
            )
        )

        opening_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
        )

        closing_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9)
            )
        )

        dilation_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (7, 7)
            )
        )

        # -------------------------------------------------
        # TRACKER AND COUNTER
        # -------------------------------------------------
        tracker = CentroidTracker(
            max_disappeared=12,
            max_distance=70
        )

        line_counter = LineCrossingCounter(
            line_margin=6,
            minimum_visible_frames=3
        )

        processed_frames = 0
        maximum_visible_objects = 0

        # -------------------------------------------------
        # PROCESS EVERY FRAME
        # -------------------------------------------------
        while True:

            frame_read_successfully, frame = (
                video_capture.read()
            )

            if not frame_read_successfully:
                break

            if (
                frame.shape[1] != output_width
                or frame.shape[0] != output_height
            ):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA
                )

            blurred_frame = cv2.GaussianBlur(
                frame,
                (5, 5),
                0
            )

            foreground_mask = (
                background_subtractor.apply(
                    blurred_frame
                )
            )

            _, binary_mask = cv2.threshold(
                foreground_mask,
                200,
                255,
                cv2.THRESH_BINARY
            )

            binary_mask = cv2.medianBlur(
                binary_mask,
                5
            )

            cleaned_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                opening_kernel,
                iterations=1
            )

            cleaned_mask = cv2.morphologyEx(
                cleaned_mask,
                cv2.MORPH_CLOSE,
                closing_kernel,
                iterations=2
            )

            cleaned_mask = cv2.dilate(
                cleaned_mask,
                dilation_kernel,
                iterations=1
            )

            cleaned_mask = remove_small_regions(
                cleaned_mask,
                minimum_area=120
            )

            # Restrict detections to the road ROI
            cleaned_mask = cv2.bitwise_and(
                cleaned_mask,
                roi_mask
            )

            visible_objects = {}

            # Initial frames are used only for
            # background learning.
            if processed_frames >= warmup_frames:

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                _, detections = detect_moving_objects(
                    frame_rgb=frame_rgb,
                    motion_mask=cleaned_mask,
                    minimum_area=minimum_area
                )

                rectangles = [
                    (
                        detection["x"],
                        detection["y"],
                        detection["width"],
                        detection["height"]
                    )
                    for detection in detections
                ]

                visible_objects = tracker.update(
                    rectangles
                )

                line_counter.update(
                    visible_objects=visible_objects,
                    frame_number=processed_frames,
                    fps=fps,
                    line_y=counting_line_y
                )

            maximum_visible_objects = max(
                maximum_visible_objects,
                len(visible_objects)
            )

            # -------------------------------------------------
            # DRAW ROAD ROI
            # -------------------------------------------------
            cv2.polylines(
                frame,
                [roi_polygon],
                isClosed=True,
                color=(255, 180, 0),
                thickness=2
            )

            # -------------------------------------------------
            # DRAW COUNTING LINE
            # -------------------------------------------------
            cv2.line(
                frame,
                (0, counting_line_y),
                (output_width - 1, counting_line_y),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "Counting Line",
                (15, max(counting_line_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # DRAW TRACKED OBJECTS
            # -------------------------------------------------
            for object_id, object_data in visible_objects.items():

                x, y, width, height = (
                    object_data["box"]
                )

                center_x, center_y = (
                    object_data["centroid"]
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (
                        int(center_x),
                        int(center_y)
                    ),
                    4,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"ID {object_id}",
                    (x, max(y - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # -------------------------------------------------
            # DRAW LIVE STATISTICS
            # -------------------------------------------------
            overlay_lines = [
                (
                    f"Frame: {processed_frames + 1}"
                    f"/{total_frames}"
                ),
                (
                    f"Visible objects: "
                    f"{len(visible_objects)}"
                ),
                (
                    f"Total crossings: "
                    f"{line_counter.total_count}"
                ),
                (
                    f"Downward: "
                    f"{line_counter.downward_count}"
                ),
                (
                    f"Upward: "
                    f"{line_counter.upward_count}"
                )
            ]

            text_y = 28

            for text in overlay_lines:

                cv2.putText(
                    frame,
                    text,
                    (15, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                text_y += 27

            if processed_frames < warmup_frames:

                cv2.putText(
                    frame,
                    "Learning background...",
                    (15, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            video_writer.write(frame)

            processed_frames += 1

            if (
                progress_callback is not None
                and (
                    processed_frames % 5 == 0
                    or processed_frames == total_frames
                )
            ):
                progress_fraction = min(
                    processed_frames / total_frames,
                    1.0
                )

                progress_callback(
                    progress_fraction,
                    processed_frames,
                    total_frames
                )

        return {
            "source_video": str(input_video_path),
            "output_path": str(output_path),
            "processed_frames": processed_frames,
            "total_registered_ids": (
                tracker.next_object_id - 1
            ),
            "maximum_visible_objects": (
                maximum_visible_objects
            ),
            "total_count": (
                line_counter.total_count
            ),
            "downward_count": (
                line_counter.downward_count
            ),
            "upward_count": (
                line_counter.upward_count
            ),
            "crossing_events": (
                line_counter.crossing_events
            ),
            "fps": fps,
            "output_width": output_width,
            "output_height": output_height,
            "duration_seconds": (
                processed_frames / fps
            ),
            "counting_line_y": counting_line_y,
            "roi_top_y": roi_top_y
        }

    finally:
        video_capture.release()

        if video_writer is not None:
            video_writer.release()