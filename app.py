"""
Vehicle Counting & Classification — Gradio App
================================================
Real-time vehicle detection, classification, and counting using
YOLOv8 + OpenCV with an intelligent mock fallback for demo purposes.

Run:  python app.py
Visit: http://localhost:7860
"""

import gradio as gr
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import os

from utils.detector import VehicleDetector, draw_detections, compute_class_distribution, VEHICLE_COLORS
from utils.tracker import CentroidTracker
from utils.counter import LineCounter
from assets.sample_frame import generate_traffic_frame, generate_sample_video_frames

# ── Global state ──────────────────────────────────────────────────────────────
detector = VehicleDetector(use_mock=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def make_distribution_chart(dist: dict) -> np.ndarray:
    """Create a styled bar chart of vehicle class distribution."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    classes = list(dist.keys())
    counts = list(dist.values())
    colors_rgb = [
        tuple(c / 255 for c in VEHICLE_COLORS.get(cls, (200, 200, 200))[::-1])
        for cls in classes
    ]

    bars = ax.bar(classes, counts, color=colors_rgb, edgecolor="white",
                  linewidth=0.5, width=0.6)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(count), ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color="white")

    ax.set_xlabel("Vehicle Class", fontsize=10, color="#aaa", labelpad=8)
    ax.set_ylabel("Count", fontsize=10, color="#aaa", labelpad=8)
    ax.set_title("Vehicle Distribution", fontsize=13, fontweight="bold",
                 color="white", pad=12)
    ax.tick_params(colors="#888", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    chart_img = np.array(Image.open(buf))
    return chart_img


def build_stats_html(dist: dict, total: int) -> str:
    """Build an HTML stats card."""
    cards = ""
    for cls_name, count in dist.items():
        color = VEHICLE_COLORS.get(cls_name, (200, 200, 200))
        hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        pct = (count / total * 100) if total > 0 else 0
        cards += f"""
        <div style="background: linear-gradient(135deg, {hex_color}22, {hex_color}44);
                    border-left: 3px solid {hex_color}; border-radius: 8px;
                    padding: 10px 14px; min-width: 100px; text-align: center;">
            <div style="font-size: 22px; font-weight: 700; color: {hex_color};">{count}</div>
            <div style="font-size: 11px; color: #aaa; text-transform: uppercase;
                        letter-spacing: 1px; margin-top: 2px;">{cls_name}</div>
            <div style="font-size: 10px; color: #666; margin-top: 2px;">{pct:.0f}%</div>
        </div>"""

    return f"""
    <div style="background: #0f1117; border-radius: 12px; padding: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
            <span style="font-size: 20px;">🚗</span>
            <span style="font-size: 16px; font-weight: 600; color: white;">
                {total} Vehicles Detected
            </span>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">{cards}</div>
    </div>"""


# ── Tab 1: Image Detection ───────────────────────────────────────────────────

def detect_image(image, confidence, use_mock):
    """Process a single image for vehicle detection."""
    if image is None:
        # Use sample frame
        image = generate_traffic_frame(seed=42)
    else:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.shape[2] == 4:  # RGBA → RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    det = VehicleDetector(use_mock=use_mock)
    detections = det.detect(image, confidence=confidence)
    annotated = draw_detections(image, detections)
    dist = compute_class_distribution(detections)
    total = sum(dist.values())

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    chart = make_distribution_chart(dist)
    stats_html = build_stats_html(dist, total)

    return annotated_rgb, chart, stats_html


# ── Tab 2: Video Counting ────────────────────────────────────────────────────

def count_video(video_path, confidence, line_position, use_mock):
    """Process a video for vehicle counting with line crossing."""
    tracker = CentroidTracker(max_disappeared=12, max_distance=80)
    counter = LineCounter(line_position=line_position / 100.0)
    det = VehicleDetector(use_mock=use_mock)

    if video_path is None:
        # Use synthetic video
        frames = generate_sample_video_frames(num_frames=80, width=640, height=480)
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        return None, "No frames found", ""

    # Process frames
    annotated_frames = []
    for frame in frames:
        detections = det.detect(frame, confidence=confidence)
        tracker.update(detections)
        counter.update(tracker.objects, tracker.class_names, frame.shape)

        annotated = draw_detections(frame, detections)
        annotated = counter.draw_line(annotated)

        # Add frame counter overlay
        cv2.putText(annotated, f"Tracked: {tracker.get_active_count()}",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"Total Crossed: {counter.total_count}",
                     (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        annotated_frames.append(annotated)

    # Write output video
    tmp_path = os.path.join(tempfile.gettempdir(), "counted_output.mp4")
    h, w = annotated_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 15, (w, h))
    for f in annotated_frames:
        writer.write(f)
    writer.release()

    # Build count table
    summary = counter.get_summary()
    table_rows = counter.get_count_table()
    table_md = "| Vehicle Type | ↑ Up | ↓ Down | Total |\n|---|---|---|---|\n"
    for row in table_rows:
        table_md += f"| {row['Vehicle Type']} | {row['↑ Up']} | {row['↓ Down']} | {row['Total']} |\n"

    stats_html = f"""
    <div style="background: #0f1117; border-radius: 12px; padding: 16px; color: white;">
        <h3 style="margin: 0 0 10px 0;">📊 Counting Summary</h3>
        <p>Processed <strong>{len(frames)}</strong> frames</p>
        <p>Total vehicles crossed: <strong style="color: #4ecdc4; font-size: 24px;">
            {counter.total_count}</strong></p>
        <p>Unique tracked: <strong>{tracker.next_id}</strong></p>
    </div>"""

    return tmp_path, stats_html, table_md


# ── Tab 3: Live Demo ─────────────────────────────────────────────────────────

def run_live_demo():
    """Run a complete demo with the synthetic traffic scene."""
    frame = generate_traffic_frame(width=1280, height=720, seed=12345)
    detections = detector.detect(frame, confidence=0.5)
    annotated = draw_detections(frame, detections)
    dist = compute_class_distribution(detections)
    total = sum(dist.values())

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    chart = make_distribution_chart(dist)
    stats_html = build_stats_html(dist, total)

    return annotated_rgb, chart, stats_html


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    block_background_fill="#111118",
    block_background_fill_dark="#111118",
    block_border_width="0px",
    block_shadow="0 4px 24px rgba(0,0,0,0.3)",
    input_background_fill="#1a1a24",
    input_background_fill_dark="#1a1a24",
    button_primary_background_fill="linear-gradient(135deg, #06b6d4, #10b981)",
    button_primary_background_fill_hover="linear-gradient(135deg, #22d3ee, #34d399)",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
.gr-button-primary { border-radius: 10px !important; font-weight: 600 !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="Vehicle Counting & Classification") as app:
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="background: linear-gradient(135deg, #06b6d4, #10b981);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.2rem; margin: 0; font-weight: 800;">
            🚗 Vehicle Counting & Classification
        </h1>
        <p style="color: #888; margin-top: 6px; font-size: 0.95rem;">
            Real-time vehicle detection, tracking & counting powered by YOLOv8 + OpenCV
        </p>
    </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Image Detection ────────────────────────────────────────
        with gr.Tab("🖼️ Image Detection", id="image_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Image", type="pil",
                                         height=350)
                    conf_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                            label="Confidence Threshold")
                    mock_toggle = gr.Checkbox(value=True, label="Mock Mode (no model required)")
                    detect_btn = gr.Button("🔍 Detect Vehicles", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Detection Result", height=350)
                    stats_output = gr.HTML(label="Statistics")
                    chart_output = gr.Image(label="Distribution Chart")

            detect_btn.click(
                detect_image,
                inputs=[img_input, conf_slider, mock_toggle],
                outputs=[img_output, chart_output, stats_output],
            )

        # ── Tab 2: Video Counting ─────────────────────────────────────────
        with gr.Tab("🎬 Video Counting", id="video_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video (or leave empty for demo)")
                    vid_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                         label="Confidence Threshold")
                    line_pos = gr.Slider(10, 90, value=50, step=1,
                                         label="Counting Line Position (%)")
                    vid_mock = gr.Checkbox(value=True, label="Mock Mode")
                    count_btn = gr.Button("📊 Count Vehicles", variant="primary", size="lg")

                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Counted Video")
                    vid_stats = gr.HTML(label="Counting Summary")
                    vid_table = gr.Markdown(label="Count Table")

            count_btn.click(
                count_video,
                inputs=[vid_input, vid_conf, line_pos, vid_mock],
                outputs=[vid_output, vid_stats, vid_table],
            )

        # ── Tab 3: Live Demo ──────────────────────────────────────────────
        with gr.Tab("🎯 Live Demo", id="demo_tab"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #06b6d422, #10b98122);
                        border-radius: 12px; padding: 16px; margin-bottom: 16px;
                        border: 1px solid #06b6d444;">
                <p style="color: #ccc; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #22d3ee;">💡 Demo Mode</strong> — Click the button
                    below to run detection on a synthetically generated traffic scene.
                    No image upload or model download required.
                </p>
            </div>
            """)
            demo_btn = gr.Button("🚀 Run Live Demo", variant="primary", size="lg")

            with gr.Row():
                demo_img = gr.Image(label="Detection Result", height=400)
                with gr.Column():
                    demo_stats = gr.HTML(label="Statistics")
                    demo_chart = gr.Image(label="Distribution")

            demo_btn.click(run_live_demo, outputs=[demo_img, demo_chart, demo_stats])

    gr.HTML("""
    <div style="text-align: center; padding: 16px; color: #555; font-size: 0.8rem;">
        Built with Gradio • OpenCV • YOLOv8 | © 2024 William Surya
    </div>
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
