"""
Synthetic traffic scene generator for demo purposes.
Creates realistic-looking road scenes with vehicle shapes.
"""

import cv2
import numpy as np
from typing import Tuple


def generate_traffic_frame(width: int = 1280, height: int = 720,
                           seed: int = None) -> np.ndarray:
    """
    Generate a synthetic traffic scene image.

    Creates a top-down view of a road with lane markings,
    sidewalks, and some environmental details.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        seed: Random seed for reproducibility

    Returns:
        BGR image as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top portion)
    sky_height = int(height * 0.25)
    for y in range(sky_height):
        ratio = y / sky_height
        r = int(40 + ratio * 60)
        g = int(60 + ratio * 80)
        b = int(120 + ratio * 80)
        frame[y, :] = (b, g, r)

    # Ground / grass areas
    frame[sky_height:, :] = (45, 85, 35)  # Dark green

    # Road surface
    road_y1 = int(height * 0.30)
    road_y2 = int(height * 0.95)
    road_x1 = int(width * 0.15)
    road_x2 = int(width * 0.85)

    # Main road (dark asphalt)
    cv2.rectangle(frame, (road_x1, road_y1), (road_x2, road_y2), (55, 55, 60), -1)

    # Road texture noise
    noise = np.random.randint(-8, 8, (road_y2 - road_y1, road_x2 - road_x1, 3), dtype=np.int16)
    road_region = frame[road_y1:road_y2, road_x1:road_x2].astype(np.int16)
    road_region = np.clip(road_region + noise, 0, 255).astype(np.uint8)
    frame[road_y1:road_y2, road_x1:road_x2] = road_region

    # Sidewalks
    sidewalk_w = int(width * 0.04)
    cv2.rectangle(frame, (road_x1 - sidewalk_w, road_y1),
                  (road_x1, road_y2), (140, 140, 130), -1)
    cv2.rectangle(frame, (road_x2, road_y1),
                  (road_x2 + sidewalk_w, road_y2), (140, 140, 130), -1)

    # Lane markings (dashed center line)
    center_x = (road_x1 + road_x2) // 2
    dash_length = 40
    gap_length = 25
    y = road_y1 + 20
    while y < road_y2 - 20:
        cv2.line(frame, (center_x, y), (center_x, min(y + dash_length, road_y2 - 20)),
                 (180, 180, 180), 2, cv2.LINE_AA)
        y += dash_length + gap_length

    # Edge lines (solid white)
    cv2.line(frame, (road_x1 + 5, road_y1), (road_x1 + 5, road_y2),
             (200, 200, 200), 2, cv2.LINE_AA)
    cv2.line(frame, (road_x2 - 5, road_y1), (road_x2 - 5, road_y2),
             (200, 200, 200), 2, cv2.LINE_AA)

    # Draw some synthetic vehicle shapes
    vehicle_colors = [
        (180, 50, 50), (50, 50, 180), (50, 150, 50),
        (200, 200, 60), (160, 80, 160), (60, 180, 180),
        (200, 120, 60), (100, 100, 180), (180, 180, 180),
    ]

    num_vehicles = np.random.randint(6, 12)
    for _ in range(num_vehicles):
        vw = np.random.randint(35, 80)
        vh = np.random.randint(25, 55)
        vx = np.random.randint(road_x1 + 20, road_x2 - vw - 20)
        vy = np.random.randint(road_y1 + 20, road_y2 - vh - 20)
        color = vehicle_colors[np.random.randint(len(vehicle_colors))]

        # Vehicle body
        cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), color, -1)

        # Windshield
        ws_y1 = vy + 3
        ws_y2 = vy + int(vh * 0.35)
        ws_x1 = vx + int(vw * 0.1)
        ws_x2 = vx + int(vw * 0.9)
        cv2.rectangle(frame, (ws_x1, ws_y1), (ws_x2, ws_y2), (100, 90, 80), -1)

        # Shadow
        shadow_pts = np.array([
            [vx + vw, vy + int(vh * 0.3)],
            [vx + vw + 8, vy + int(vh * 0.4)],
            [vx + vw + 8, vy + vh + 5],
            [vx + vw, vy + vh],
        ])
        cv2.fillPoly(frame, [shadow_pts], (30, 30, 35))

    # Add some ambient lighting / vignette
    vignette = np.zeros_like(frame, dtype=np.float32)
    cy, cx = height // 2, width // 2
    for y in range(height):
        for x in range(0, width, 4):  # Step for performance
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            factor = 1.0 - 0.3 * (dist / max_dist) ** 2
            vignette[y, x:min(x+4, width)] = factor

    frame = (frame.astype(np.float32) * vignette).astype(np.uint8)

    return frame


def generate_sample_video_frames(num_frames: int = 60, width: int = 640,
                                  height: int = 480) -> list:
    """
    Generate a sequence of frames simulating moving vehicles.

    Args:
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height

    Returns:
        List of BGR frames
    """
    frames = []
    road_x1 = int(width * 0.15)
    road_x2 = int(width * 0.85)
    road_y1 = int(height * 0.25)
    road_y2 = int(height * 0.95)

    # Define moving vehicles with trajectories
    vehicles = []
    num_vehicles = np.random.randint(4, 8)

    vehicle_colors = [
        (180, 50, 50), (50, 50, 180), (50, 150, 50),
        (200, 200, 60), (160, 80, 160),
    ]

    for i in range(num_vehicles):
        lane_x = np.random.randint(road_x1 + 30, road_x2 - 60)
        start_y = np.random.randint(-100, -20)
        speed = np.random.uniform(3, 8)
        vw = np.random.randint(30, 60)
        vh = np.random.randint(22, 45)
        color = vehicle_colors[i % len(vehicle_colors)]
        vehicles.append({
            "x": lane_x, "y": float(start_y), "speed": speed,
            "w": vw, "h": vh, "color": color
        })

    for frame_idx in range(num_frames):
        # Base road
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:road_y1, :] = (60, 80, 40)  # Grass above road
        frame[road_y1:road_y2, :] = (55, 55, 60)  # Road
        frame[road_y2:, :] = (60, 80, 40)  # Grass below road

        # Lane markings
        center_x = width // 2
        y = road_y1 + (frame_idx * 3) % 65
        while y < road_y2:
            cv2.line(frame, (center_x, y), (center_x, min(y + 40, road_y2)),
                     (180, 180, 180), 2)
            y += 65

        # Draw and update vehicles
        for v in vehicles:
            v["y"] += v["speed"]
            if v["y"] > height + 50:
                v["y"] = np.random.randint(-150, -30)
                v["x"] = np.random.randint(road_x1 + 30, road_x2 - 60)

            vy = int(v["y"])
            if -v["h"] < vy < height:
                cv2.rectangle(frame, (v["x"], vy),
                              (v["x"] + v["w"], vy + v["h"]),
                              v["color"], -1)

        frames.append(frame)

    return frames
