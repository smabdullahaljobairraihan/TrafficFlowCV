from collections import defaultdict


class LineCrossingCounter:
    """
    Count tracked objects when their centroids cross
    a horizontal counting line.
    """

    def __init__(
        self,
        line_margin=6,
        minimum_visible_frames=3
    ):
        self.line_margin = line_margin

        self.minimum_visible_frames = (
            minimum_visible_frames
        )

        self.last_side = {}
        self.visible_frame_counts = defaultdict(int)

        self.counted_ids = set()
        self.crossing_events = []

        self.upward_count = 0
        self.downward_count = 0

    def get_side(self, center_y, line_y):
        """
        Determine whether a centroid is above, below,
        or inside the counting-line tolerance zone.
        """

        if center_y < line_y - self.line_margin:
            return -1

        if center_y > line_y + self.line_margin:
            return 1

        return 0

    def update(
        self,
        visible_objects,
        frame_number,
        fps,
        line_y
    ):
        """
        Update crossing records using objects visible
        in the current video frame.
        """

        new_events = []

        for object_id, object_data in visible_objects.items():

            center_x, center_y = (
                object_data["centroid"]
            )

            center_x = int(center_x)
            center_y = int(center_y)

            self.visible_frame_counts[object_id] += 1

            current_side = self.get_side(
                center_y,
                line_y
            )

            previous_side = self.last_side.get(
                object_id
            )

            # Only update the stored side when the centroid
            # is outside the tolerance zone.
            if current_side != 0:

                can_be_counted = (
                    previous_side is not None
                    and previous_side != current_side
                    and object_id not in self.counted_ids
                    and self.visible_frame_counts[object_id]
                    >= self.minimum_visible_frames
                )

                if can_be_counted:

                    if (
                        previous_side == -1
                        and current_side == 1
                    ):
                        direction = "Downward"
                        self.downward_count += 1

                    else:
                        direction = "Upward"
                        self.upward_count += 1

                    if fps > 0:
                        crossing_time = frame_number / fps
                    else:
                        crossing_time = 0

                    event = {
                        "object_id": object_id,
                        "frame_number": frame_number,
                        "time_seconds": round(
                            crossing_time,
                            2
                        ),
                        "direction": direction,
                        "center_x": center_x,
                        "center_y": center_y
                    }

                    self.crossing_events.append(event)
                    new_events.append(event)

                    self.counted_ids.add(object_id)

                self.last_side[object_id] = (
                    current_side
                )

        return new_events

    @property
    def total_count(self):
        """
        Return the number of unique IDs that crossed
        the counting line.
        """

        return len(self.counted_ids)