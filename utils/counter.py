"""
Line-crossing counter for vehicle counting.
Supports configurable counting lines with directional tracking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import time


@dataclass
class CountEvent:
    """A single counting event."""
    object_id: int
    class_name: str
    direction: str  # "up", "down", "left", "right"
    timestamp: float
    position: Tuple[int, int]


class LineCounter:
    """
    Virtual line-crossing counter for vehicle counting.

    Tracks objects crossing a defined line and records direction,
    class, and timestamp for each crossing event.
    """

    def __init__(self, line_position: float = 0.5, orientation: str = "horizontal"):
        """
        Args:
            line_position: Relative position of the counting line (0.0-1.0).
                          For horizontal: fraction of frame height from top.
                          For vertical: fraction of frame width from left.
            orientation: "horizontal" or "vertical"
        """
        self.line_position = line_position
        self.orientation = orientation
        self.counts: Dict[str, Dict[str, int]] = {}  # {class: {direction: count}}
        self.events: List[CountEvent] = []
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        self.counted_ids: set = set()
        self.total_count = 0

    def get_line_coords(self, frame_shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get the pixel coordinates of the counting line.

        Args:
            frame_shape: (height, width, channels)

        Returns:
            ((x1, y1), (x2, y2)) endpoints of the line
        """
        h, w = frame_shape[:2]
        if self.orientation == "horizontal":
            y = int(h * self.line_position)
            return ((0, y), (w, y))
        else:
            x = int(w * self.line_position)
            return ((x, 0), (x, h))

    def update(self, tracked_objects: OrderedDict, class_names: OrderedDict,
               frame_shape: Tuple[int, ...]) -> List[CountEvent]:
        """
        Check for line crossings and update counts.

        Args:
            tracked_objects: {id: centroid_array} from CentroidTracker
            class_names: {id: class_name} from CentroidTracker
            frame_shape: Shape of the current frame

        Returns:
            List of new CountEvent objects from this update
        """
        h, w = frame_shape[:2]
        new_events = []

        if self.orientation == "horizontal":
            line_pos = int(h * self.line_position)
        else:
            line_pos = int(w * self.line_position)

        for obj_id, centroid in tracked_objects.items():
            curr_pos = tuple(centroid.astype(int)) if isinstance(centroid, np.ndarray) else centroid

            if obj_id in self.prev_positions and obj_id not in self.counted_ids:
                prev_pos = self.prev_positions[obj_id]

                if self.orientation == "horizontal":
                    prev_val = prev_pos[1]
                    curr_val = curr_pos[1]
                else:
                    prev_val = prev_pos[0]
                    curr_val = curr_pos[0]

                # Check if object crossed the line
                if (prev_val < line_pos <= curr_val) or (prev_val > line_pos >= curr_val):
                    if self.orientation == "horizontal":
                        direction = "down" if curr_val > prev_val else "up"
                    else:
                        direction = "right" if curr_val > prev_val else "left"

                    cls_name = class_names.get(obj_id, "unknown")
                    event = CountEvent(
                        object_id=obj_id,
                        class_name=cls_name,
                        direction=direction,
                        timestamp=time.time(),
                        position=curr_pos,
                    )
                    new_events.append(event)
                    self.events.append(event)
                    self.counted_ids.add(obj_id)
                    self.total_count += 1

                    # Update per-class per-direction counts
                    if cls_name not in self.counts:
                        self.counts[cls_name] = {"up": 0, "down": 0, "left": 0, "right": 0}
                    self.counts[cls_name][direction] += 1

            self.prev_positions[obj_id] = curr_pos

        return new_events

    def get_summary(self) -> dict:
        """
        Get a summary of all counts.

        Returns:
            Dict with total count, per-class counts, and direction breakdown.
        """
        class_totals = {}
        for cls_name, dirs in self.counts.items():
            class_totals[cls_name] = sum(dirs.values())

        return {
            "total": self.total_count,
            "per_class": class_totals,
            "per_class_direction": self.counts,
            "events_count": len(self.events),
        }

    def get_count_table(self) -> List[dict]:
        """Get counts formatted as a table (list of dicts for DataFrame)."""
        rows = []
        for cls_name, dirs in self.counts.items():
            total = sum(dirs.values())
            rows.append({
                "Vehicle Type": cls_name.capitalize(),
                "↑ Up": dirs.get("up", 0),
                "↓ Down": dirs.get("down", 0),
                "← Left": dirs.get("left", 0),
                "→ Right": dirs.get("right", 0),
                "Total": total,
            })

        # Add totals row
        if rows:
            rows.append({
                "Vehicle Type": "TOTAL",
                "↑ Up": sum(r["↑ Up"] for r in rows),
                "↓ Down": sum(r["↓ Down"] for r in rows),
                "← Left": sum(r["← Left"] for r in rows),
                "→ Right": sum(r["→ Right"] for r in rows),
                "Total": self.total_count,
            })

        return rows

    def draw_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255),
                  thickness: int = 2) -> np.ndarray:
        """Draw the counting line on a frame."""
        import cv2
        annotated = frame.copy()
        pt1, pt2 = self.get_line_coords(frame.shape)
        cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw count label
        label = f"Count: {self.total_count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lx = (pt1[0] + pt2[0]) // 2 - tw // 2
        ly = pt1[1] - 15 if self.orientation == "horizontal" else pt1[1] + 30
        cv2.putText(annotated, label, (lx, ly),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        return annotated

    def reset(self):
        """Reset all counters."""
        self.counts.clear()
        self.events.clear()
        self.prev_positions.clear()
        self.counted_ids.clear()
        self.total_count = 0
