import os
from pathlib import Path

import cv2


def save_uploaded_video(
    uploaded_video,
    output_folder="outputs/uploads"
):
    """
    Save a Streamlit uploaded video inside the project folder.

    The file is saved only when it does not already exist
    or when its size has changed.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Prevent directory information from entering the filename
    safe_filename = Path(uploaded_video.name).name

    video_path = os.path.join(
        output_folder,
        safe_filename
    )

    should_save_file = (
        not os.path.exists(video_path)
        or os.path.getsize(video_path) != uploaded_video.size
    )

    if should_save_file:
        with open(video_path, "wb") as video_file:
            video_file.write(
                uploaded_video.getbuffer()
            )

    return video_path


def get_video_metadata(video_path):
    """
    Read the video's frame count, FPS, resolution,
    and duration using OpenCV.
    """

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(
            "OpenCV could not open the video."
        )

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

    frame_width = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    video_capture.release()

    if fps > 0:
        duration_seconds = total_frames / fps
    else:
        duration_seconds = 0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "duration_seconds": duration_seconds
    }


def read_video_frame(video_path, frame_number):
    """
    Read one selected video frame.

    OpenCV returns BGR images, so the frame is
    converted to RGB for Streamlit.
    """

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(
            "OpenCV could not open the video."
        )

    video_capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    frame_read_successfully, frame = (
        video_capture.read()
    )

    video_capture.release()

    if not frame_read_successfully:
        return None

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return frame_rgb