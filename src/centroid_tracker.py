from collections import OrderedDict

import numpy as np


class CentroidTracker:
    """
    Track moving objects by comparing centroid positions
    between consecutive video frames.
    """

    def __init__(
        self,
        max_disappeared=12,
        max_distance=70
    ):
        self.next_object_id = 1

        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.boxes = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box):
        """
        Register a newly detected object.
        """

        object_id = self.next_object_id

        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.boxes[object_id] = box

        self.next_object_id += 1

        return object_id

    def deregister(self, object_id):
        """
        Remove an object that has disappeared for too long.
        """

        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.boxes[object_id]

    def update(self, rectangles):
        """
        Match current bounding boxes with previously
        tracked objects.

        Parameters
        ----------
        rectangles : list
            List of bounding boxes in the form:
            (x, y, width, height)

        Returns
        -------
        visible_objects : dict
            Objects visible in the current frame.
        """

        visible_objects = {}

        # No objects detected in the current frame
        if len(rectangles) == 0:

            for object_id in list(
                self.disappeared.keys()
            ):
                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            return visible_objects

        # Calculate centroids of current detections
        input_centroids = np.zeros(
            (len(rectangles), 2),
            dtype="int"
        )

        for index, rectangle in enumerate(rectangles):

            x, y, width, height = rectangle

            center_x = x + width // 2
            center_y = y + height // 2

            input_centroids[index] = (
                center_x,
                center_y
            )

        # Register every detection if no objects exist
        if len(self.objects) == 0:

            for index, centroid in enumerate(
                input_centroids
            ):
                object_id = self.register(
                    centroid,
                    rectangles[index]
                )

                visible_objects[object_id] = {
                    "centroid": centroid,
                    "box": rectangles[index]
                }

            return visible_objects

        # Existing tracked objects
        object_ids = list(
            self.objects.keys()
        )

        object_centroids = np.array(
            list(self.objects.values())
        )

        # Calculate distances between previous and
        # current centroids using NumPy broadcasting
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis]
            - input_centroids[np.newaxis, :],
            axis=2
        )

        # Process nearest pairs first
        rows = distances.min(axis=1).argsort()

        columns = distances.argmin(
            axis=1
        )[rows]

        used_rows = set()
        used_columns = set()

        for row, column in zip(rows, columns):

            if row in used_rows:
                continue

            if column in used_columns:
                continue

            # Do not match objects that moved too far
            if (
                distances[row, column]
                > self.max_distance
            ):
                continue

            object_id = object_ids[row]

            self.objects[object_id] = (
                input_centroids[column]
            )

            self.boxes[object_id] = (
                rectangles[column]
            )

            self.disappeared[object_id] = 0

            visible_objects[object_id] = {
                "centroid": input_centroids[column],
                "box": rectangles[column]
            }

            used_rows.add(row)
            used_columns.add(column)

        # Previously tracked objects without a match
        unused_rows = set(
            range(distances.shape[0])
        ).difference(used_rows)

        for row in unused_rows:

            object_id = object_ids[row]

            self.disappeared[object_id] += 1

            if (
                self.disappeared[object_id]
                > self.max_disappeared
            ):
                self.deregister(object_id)

        # Current detections without an existing match
        unused_columns = set(
            range(distances.shape[1])
        ).difference(used_columns)

        for column in unused_columns:

            object_id = self.register(
                input_centroids[column],
                rectangles[column]
            )

            visible_objects[object_id] = {
                "centroid": input_centroids[column],
                "box": rectangles[column]
            }

        return visible_objects