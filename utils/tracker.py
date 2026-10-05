"""
Centroid-based multi-object tracker.
Uses the Hungarian algorithm (scipy) for optimal assignment.
"""

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class CentroidTracker:
    """
    Simple centroid tracker that assigns persistent IDs across frames.

    Uses Euclidean distance and the Hungarian algorithm for optimal
    detection-to-track assignment. Handles temporary occlusions with
    a configurable grace period.
    """

    def __init__(self, max_disappeared: int = 10, max_distance: float = 80.0):
        """
        Args:
            max_disappeared: Number of consecutive frames an object can be
                             missing before deregistration.
            max_distance: Maximum Euclidean distance for a valid assignment.
        """
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.class_names: OrderedDict[int, str] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories: dict[int, List[Tuple[int, int]]] = {}

    def register(self, centroid: np.ndarray, class_name: str = "unknown") -> int:
        """Register a new object and return its ID."""
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.class_names[obj_id] = class_name
        self.trajectories[obj_id] = [tuple(centroid.astype(int))]
        self.next_id += 1
        return obj_id

    def deregister(self, object_id: int):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.class_names[object_id]
        # Keep trajectory for historical reference

    def update(self, detections: list) -> OrderedDict:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects with .center and .class_name

        Returns:
            OrderedDict mapping object IDs to their centroids.
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([det.center for det in detections])
        input_classes = [det.class_name for det in detections]

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_classes[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2
        )

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(dist_matrix)

        used_rows = set()
        used_cols = set()

        for row, col in zip(row_indices, col_indices):
            if dist_matrix[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            self.class_names[obj_id] = input_classes[col]

            if obj_id in self.trajectories:
                self.trajectories[obj_id].append(tuple(input_centroids[col].astype(int)))

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Handle unmatched new detections (register)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col], input_classes[col])

        return self.objects

    def get_active_count(self) -> int:
        """Return number of currently tracked objects."""
        return len(self.objects)

    def get_class_counts(self) -> dict:
        """Return count per class for currently tracked objects."""
        counts = {}
        for cls_name in self.class_names.values():
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts

    def reset(self):
        """Reset tracker state."""
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.class_names.clear()
        self.trajectories.clear()
