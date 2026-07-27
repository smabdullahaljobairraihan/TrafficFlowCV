import cv2
import numpy as np


def create_motion_preview(
    video_path,
    target_frame_number,
    warmup_frames=150,
    resize_width=640
):
    """
    Generate a cleaned motion mask for one selected video frame.

    Parameters
    ----------
    video_path : str
        Path of the uploaded video.

    target_frame_number : int
        Frame number selected by the user.

    warmup_frames : int
        Number of earlier frames used to help MOG2
        learn the background.

    resize_width : int
        Width used during processing to improve speed.

    Returns
    -------
    final_frame_rgb : numpy.ndarray
        Selected traffic frame converted from BGR to RGB.

    final_motion_mask : numpy.ndarray
        Cleaned black-and-white foreground mask.

    processed_frame_count : int
        Number of frames processed before reaching
        the selected frame.
    """

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(
            "OpenCV could not open the video."
        )

    try:
        # -------------------------------------------------
        # READ VIDEO INFORMATION
        # -------------------------------------------------
        total_frames = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            raise ValueError(
                "The video does not contain readable frames."
            )

        # Keep the selected frame inside the valid range
        target_frame_number = max(
            0,
            min(
                target_frame_number,
                total_frames - 1
            )
        )

        # Start several frames before the selected frame
        start_frame_number = max(
            0,
            target_frame_number - warmup_frames
        )

        video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame_number
        )

        # -------------------------------------------------
        # CREATE BACKGROUND SUBTRACTOR
        # -------------------------------------------------
        background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=max(warmup_frames, 150),
                varThreshold=25,
                detectShadows=True
            )
        )

        # Small kernel removes isolated noise
        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        # Larger kernel joins nearby parts of vehicles
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )

        # Used for slight foreground expansion
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (7, 7)
        )

        current_frame_number = start_frame_number
        processed_frame_count = 0

        final_frame_rgb = None
        final_motion_mask = None

        # -------------------------------------------------
        # PROCESS FRAMES SEQUENTIALLY
        # -------------------------------------------------
        while current_frame_number <= target_frame_number:

            frame_read_successfully, frame = (
                video_capture.read()
            )

            if not frame_read_successfully:
                break

            original_height, original_width = (
                frame.shape[:2]
            )

            # Resize large frames for faster processing
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
                    (resize_width, resized_height),
                    interpolation=cv2.INTER_AREA
                )

            # Reduce small pixel and compression changes
            blurred_frame = cv2.GaussianBlur(
                frame,
                (5, 5),
                0
            )

            # Generate foreground mask
            foreground_mask = (
                background_subtractor.apply(
                    blurred_frame
                )
            )

            # MOG2 commonly produces:
            # 0   = background
            # 127 = shadow
            # 255 = foreground
            #
            # Thresholding above 200 removes shadows.
            _, binary_mask = cv2.threshold(
                foreground_mask,
                200,
                255,
                cv2.THRESH_BINARY
            )

            # Remove salt-and-pepper noise
            binary_mask = cv2.medianBlur(
                binary_mask,
                5
            )

            # Remove small isolated foreground pixels
            cleaned_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                opening_kernel,
                iterations=1
            )

            # Join fragmented sections of vehicles
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask,
                cv2.MORPH_CLOSE,
                closing_kernel,
                iterations=2
            )

            # Expand vehicle regions slightly
            cleaned_mask = cv2.dilate(
                cleaned_mask,
                dilation_kernel,
                iterations=1
            )

            # Remove very small connected regions
            cleaned_mask = remove_small_regions(
                cleaned_mask,
                minimum_area=120
            )

            # Save the latest processed frame
            final_frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            final_motion_mask = cleaned_mask

            processed_frame_count += 1
            current_frame_number += 1

        if (
            final_frame_rgb is None
            or final_motion_mask is None
        ):
            raise ValueError(
                "The selected frame could not be processed."
            )

        return (
            final_frame_rgb,
            final_motion_mask,
            processed_frame_count
        )

    finally:
        video_capture.release()


def remove_small_regions(
    binary_mask,
    minimum_area=120
):
    """
    Remove small connected white regions from a binary mask.

    This reduces noise caused by headlights, reflections,
    compression artifacts, and small background movements.
    """

    number_of_labels, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    cleaned_mask = np.zeros_like(
        binary_mask
    )

    # Label zero represents the black background,
    # so processing starts from label one.
    for label_number in range(
        1,
        number_of_labels
    ):
        region_area = statistics[
            label_number,
            cv2.CC_STAT_AREA
        ]

        if region_area >= minimum_area:
            cleaned_mask[
                labels == label_number
            ] = 255

    return cleaned_mask