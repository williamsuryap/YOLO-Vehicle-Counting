"""
Vehicle Detector module with YOLO integration and mock fallback.
Supports YOLOv8 when ultralytics is installed, otherwise falls back
to procedural mock detections for demo purposes.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


VEHICLE_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "bicycle",
}

VEHICLE_COLORS = {
    "car": (78, 172, 248),       # Blue
    "truck": (120, 230, 150),    # Green
    "bus": (255, 180, 60),       # Orange
    "motorcycle": (220, 100, 255), # Purple
    "bicycle": (100, 255, 255),  # Cyan
}

# COCO class IDs that map to vehicles
COCO_VEHICLE_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}


@dataclass
class Detection:
    """A single vehicle detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str
    confidence: float
    class_id: int = 0
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class VehicleDetector:
    """
    Vehicle detector with automatic fallback.
    Uses YOLOv8 if ultralytics is available, otherwise generates mock detections.
    """

    def __init__(self, model_path: str = "yolov8n.pt", use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None
        self.model_path = model_path

        if not use_mock:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.use_mock = False
            except ImportError:
                print("[WARNING] ultralytics not installed. Falling back to mock detector.")
                self.use_mock = True

    def detect(self, frame: np.ndarray, confidence: float = 0.5) -> List[Detection]:
        """
        Detect vehicles in a frame.

        Args:
            frame: BGR image as numpy array
            confidence: Minimum confidence threshold

        Returns:
            List of Detection objects
        """
        if self.use_mock:
            return self._mock_detect(frame, confidence)
        return self._yolo_detect(frame, confidence)

    def _yolo_detect(self, frame: np.ndarray, confidence: float) -> List[Detection]:
        """Run real YOLOv8 inference."""
        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in COCO_VEHICLE_MAP:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_name=COCO_VEHICLE_MAP[cls_id],
                        confidence=conf,
                        class_id=cls_id,
                    ))

        return detections

    def _mock_detect(self, frame: np.ndarray, confidence: float) -> List[Detection]:
        """Generate realistic mock detections for demo purposes."""
        h, w = frame.shape[:2]
        rng = np.random.RandomState(hash(frame.tobytes()[:1000]) % (2**31))
        num_vehicles = rng.randint(4, 14)

        detections = []
        road_y_start = int(h * 0.3)
        road_y_end = int(h * 0.95)

        class_weights = [0.45, 0.15, 0.10, 0.20, 0.10]  # car, truck, bus, moto, bicycle
        class_ids = list(VEHICLE_CLASSES.keys())

        for _ in range(num_vehicles):
            cls_id = rng.choice(class_ids, p=class_weights)
            cls_name = VEHICLE_CLASSES[cls_id]

            # Size varies by vehicle type
            if cls_name == "truck":
                vw = rng.randint(int(w * 0.10), int(w * 0.18))
                vh = rng.randint(int(h * 0.08), int(h * 0.14))
            elif cls_name == "bus":
                vw = rng.randint(int(w * 0.12), int(w * 0.22))
                vh = rng.randint(int(h * 0.08), int(h * 0.13))
            elif cls_name in ("motorcycle", "bicycle"):
                vw = rng.randint(int(w * 0.03), int(w * 0.07))
                vh = rng.randint(int(h * 0.04), int(h * 0.08))
            else:  # car
                vw = rng.randint(int(w * 0.07), int(w * 0.13))
                vh = rng.randint(int(h * 0.05), int(h * 0.10))

            cx = rng.randint(vw // 2, w - vw // 2)
            cy = rng.randint(road_y_start + vh // 2, road_y_end - vh // 2)

            x1 = max(0, cx - vw // 2)
            y1 = max(0, cy - vh // 2)
            x2 = min(w, cx + vw // 2)
            y2 = min(h, cy + vh // 2)

            conf = round(rng.uniform(max(confidence, 0.65), 0.99), 2)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_name=cls_name,
                confidence=conf,
                class_id=cls_id,
            ))

        return detections


def draw_detections(frame: np.ndarray, detections: List[Detection],
                    show_confidence: bool = True) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on a frame.

    Args:
        frame: BGR image
        detections: List of Detection objects
        show_confidence: Whether to display confidence scores

    Returns:
        Annotated frame copy
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = VEHICLE_COLORS.get(det.class_name, (200, 200, 200))

        # Draw filled rectangle background for label
        label = det.class_name.upper()
        if show_confidence:
            label += f" {det.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

        # Label text
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def compute_class_distribution(detections: List[Detection]) -> dict:
    """Compute count per vehicle class."""
    dist = {name: 0 for name in VEHICLE_CLASSES.values()}
    for det in detections:
        if det.class_name in dist:
            dist[det.class_name] += 1
    return dist
