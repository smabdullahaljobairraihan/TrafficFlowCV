import cv2


def detect_moving_objects(
    frame_rgb,
    motion_mask,
    minimum_area=300,
    minimum_width=15,
    minimum_height=12
):
    """
    Detect candidate moving objects from a binary motion mask.

    Parameters
    ----------
    frame_rgb : numpy.ndarray
        RGB video frame on which bounding boxes will be drawn.

    motion_mask : numpy.ndarray
        Binary mask containing white moving regions.

    minimum_area : int
        Minimum contour area required for detection.

    minimum_width : int
        Minimum bounding-box width.

    minimum_height : int
        Minimum bounding-box height.

    Returns
    -------
    annotated_frame : numpy.ndarray
        RGB frame with bounding boxes and labels.

    detections : list
        Information about every accepted moving region.
    """

    # Copy the frame so the original frame is unchanged
    annotated_frame = frame_rgb.copy()

    # Find the external boundaries of white regions
    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        contour_area = cv2.contourArea(contour)

        # Ignore very small regions
        if contour_area < minimum_area:
            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        # Ignore regions that are too narrow or too short
        if (
            width < minimum_width
            or height < minimum_height
        ):
            continue

        aspect_ratio = width / height

        # Remove extremely thin or unusually wide regions
        if aspect_ratio < 0.25 or aspect_ratio > 5.0:
            continue

        # Calculate the center of the bounding box
        center_x = x + width // 2
        center_y = y + height // 2

        detection_number = len(detections) + 1

        detection = {
            "detection_id": detection_number,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "area": round(contour_area, 2)
        }

        detections.append(detection)

        # Draw a bounding box
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Draw the centroid
        cv2.circle(
            annotated_frame,
            (center_x, center_y),
            4,
            (255, 0, 0),
            -1
        )

        # Add a candidate-object label
        label = f"Object {detection_number}"

        cv2.putText(
            annotated_frame,
            label,
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return annotated_frame, detections