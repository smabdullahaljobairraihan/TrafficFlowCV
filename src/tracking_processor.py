import cv2

from src.centroid_tracker import CentroidTracker
from src.motion_detector import remove_small_regions
from src.object_detector import detect_moving_objects


def create_tracking_preview(
    video_path,
    target_frame_number,
    minimum_area=300,
    tracking_frames=90,
    background_warmup_frames=120,
    resize_width=640
):
    """
    Process consecutive frames and assign persistent
    object IDs using centroid tracking.
    """

    video_capture = cv2.VideoCapture(
        video_path
    )

    if not video_capture.isOpened():
        raise ValueError(
            "OpenCV could not open the video."
        )

    try:
        total_frames = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            raise ValueError(
                "The video contains no readable frames."
            )

        target_frame_number = max(
            0,
            min(
                target_frame_number,
                total_frames - 1
            )
        )

        # Frames in this interval will be tracked
        tracking_start_frame = max(
            0,
            target_frame_number
            - tracking_frames
            + 1
        )

        # Earlier frames are used only to learn background
        processing_start_frame = max(
            0,
            tracking_start_frame
            - background_warmup_frames
        )

        video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            processing_start_frame
        )

        background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=max(
                    background_warmup_frames,
                    150
                ),
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

        tracker = CentroidTracker(
            max_disappeared=12,
            max_distance=70
        )

        current_frame_number = (
            processing_start_frame
        )

        processed_frame_count = 0

        final_annotated_frame = None
        final_motion_mask = None
        final_visible_count = 0

        while (
            current_frame_number
            <= target_frame_number
        ):

            frame_read_successfully, frame = (
                video_capture.read()
            )

            if not frame_read_successfully:
                break

            original_height, original_width = (
                frame.shape[:2]
            )

            if (
                resize_width is not None
                and original_width > resize_width
            ):
                resize_ratio = (
                    resize_width / original_width
                )

                resized_height = int(
                    original_height * resize_ratio
                )

                frame = cv2.resize(
                    frame,
                    (
                        resize_width,
                        resized_height
                    ),
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

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Only begin tracking after background warmup
            if (
                current_frame_number
                >= tracking_start_frame
            ):
                _, detections = (
                    detect_moving_objects(
                        frame_rgb=frame_rgb,
                        motion_mask=cleaned_mask,
                        minimum_area=minimum_area
                    )
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

                annotated_frame = (
                    frame_rgb.copy()
                )

                for (
                    object_id,
                    object_data
                ) in visible_objects.items():

                    x, y, width, height = (
                        object_data["box"]
                    )

                    center_x, center_y = (
                        object_data["centroid"]
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x, y),
                        (
                            x + width,
                            y + height
                        ),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        annotated_frame,
                        (
                            int(center_x),
                            int(center_y)
                        ),
                        4,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        annotated_frame,
                        f"ID {object_id}",
                        (
                            x,
                            max(y - 8, 15)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                final_annotated_frame = (
                    annotated_frame
                )

                final_visible_count = len(
                    visible_objects
                )

            final_motion_mask = cleaned_mask

            processed_frame_count += 1
            current_frame_number += 1

        if final_annotated_frame is None:
            raise ValueError(
                "Tracking preview could not be generated."
            )

        total_registered_ids = (
            tracker.next_object_id - 1
        )

        return {
            "annotated_frame": final_annotated_frame,
            "motion_mask": final_motion_mask,
            "visible_objects": final_visible_count,
            "registered_ids": total_registered_ids,
            "processed_frames": processed_frame_count,
            "tracking_start_frame": tracking_start_frame,
            "target_frame": target_frame_number
        }

    finally:
        video_capture.release()