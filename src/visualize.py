"""
Visualizer for Neural Gesture Mannequin:
Mode 1: Skeleton-Based Model (Kinematic Wireframe Rig)
Mode 2: Volume-Based Model (3D Tapered Cylinders, Cross-Contour Ellipses & Torso Box)
"""

import math
from typing import List, Tuple
import cv2
import numpy as np
import torch

from src.dataset import COCO_SKELETON_EDGES, SKELETON_COLORS


def draw_skeleton_mode(
    canvas: np.ndarray,
    coords: np.ndarray,
    vis: np.ndarray,
    conf_thresh: float = 0.3
) -> np.ndarray:
    """
    Renders Mode 1: Graph-based Skeleton Wireframe (Nodes & Bones).
    """
    out = canvas.copy()
    h, w, _ = out.shape

    # 1. Draw connecting kinematic bones
    for idx, (u, v) in enumerate(COCO_SKELETON_EDGES):
        if vis[u] > conf_thresh and vis[v] > conf_thresh:
            pt1 = (int(coords[u, 0] * w), int(coords[u, 1] * h))
            pt2 = (int(coords[v, 0] * w), int(coords[v, 1] * h))
            color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
            cv2.line(out, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)

    # 2. Draw joint nodes
    for i in range(len(coords)):
        if vis[i] > conf_thresh:
            pt = (int(coords[i, 0] * w), int(coords[i, 1] * h))
            cv2.circle(out, pt, radius=5, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(out, pt, radius=3, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return out


def draw_cylinder_with_ellipses(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    r1: float,
    r2: float,
    color: Tuple[int, int, int] = (40, 40, 40),
    num_rings: int = 3
):
    """
    Renders an anatomical 3D tapered cylinder with cross-contour ellipses.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 5:
        return

    # Perpendicular unit vector
    ux = -dy / length
    uy = dx / length

    # 4 corners of tapered cylinder
    p1_a = (int(x1 + ux * r1), int(y1 + uy * r1))
    p1_b = (int(x1 - ux * r1), int(y1 - uy * r1))
    p2_a = (int(x2 + ux * r2), int(y2 + uy * r2))
    p2_b = (int(x2 - ux * r2), int(y2 - uy * r2))

    # Outer boundary lines
    cv2.line(canvas, p1_a, p2_a, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(canvas, p1_b, p2_b, color, thickness=2, lineType=cv2.LINE_AA)

    # Cross-contour ellipses along cylinder body
    angle = math.degrees(math.atan2(dy, dx))
    for step in range(num_rings + 1):
        t = step / max(1, num_rings)
        cx = int(x1 + dx * t)
        cy = int(y1 + dy * t)
        curr_r = int(r1 * (1.0 - t) + r2 * t)
        axes = (max(2, curr_r), max(2, int(curr_r * 0.45)))
        cv2.ellipse(canvas, (cx, cy), axes, angle, 0, 360, color, thickness=1, lineType=cv2.LINE_AA)


def draw_volume_mode(
    canvas: np.ndarray,
    coords: np.ndarray,
    volume_params: np.ndarray,
    vis: np.ndarray,
    conf_thresh: float = 0.3
) -> np.ndarray:
    """
    Renders Mode 2: Volumetric 3D Anatomical Mannequin (Fig. 2 Right).
    Constructs volumetric cylinders, cross-contour ellipses, and torso boxes.
    """
    out = canvas.copy()
    h, w, _ = out.shape
    c_line = (30, 30, 30)

    # Extract predicted radii in pixels
    head_r = max(8, int(volume_params[0] * w))
    torso_w = max(16, int(volume_params[1] * w))
    uarm_r = max(6, int(volume_params[2] * w))
    farm_r = max(5, int(volume_params[3] * w))
    thigh_r = max(10, int(volume_params[4] * w))
    calf_r = max(8, int(volume_params[5] * w))

    def get_pt(idx):
        return (int(coords[idx, 0] * w), int(coords[idx, 1] * h))

    # 1. Head Sphere with Facial Crosshairs
    if vis[0] > conf_thresh or (vis[1] > conf_thresh and vis[2] > conf_thresh):
        if vis[0] > conf_thresh:
            head_center = get_pt(0)
        else:
            p1, p2 = get_pt(1), get_pt(2)
            head_center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        cv2.circle(out, head_center, head_r, c_line, thickness=2, lineType=cv2.LINE_AA)
        # Eye-line & Centerline crosshair
        cv2.ellipse(out, head_center, (head_r, int(head_r * 0.35)), 0, 0, 360, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.line(out, (head_center[0], head_center[1] - head_r), (head_center[0], head_center[1] + head_r), (90, 90, 90), 1, cv2.LINE_AA)

    # 2. Torso Box & Ribcage
    if vis[5] > conf_thresh and vis[6] > conf_thresh:
        ls, rs = get_pt(5), get_pt(6)
        lh = get_pt(11) if vis[11] > conf_thresh else (ls[0], ls[1] + int(h * 0.3))
        rh = get_pt(12) if vis[12] > conf_thresh else (rs[0], rs[1] + int(h * 0.3))

        torso_poly = np.array([ls, rs, rh, lh], dtype=np.int32)
        cv2.polylines(out, [torso_poly], isClosed=True, color=c_line, thickness=2, lineType=cv2.LINE_AA)

        # Core spine & ribcage contour ring
        mid_shoulder = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        cv2.line(out, mid_shoulder, mid_hip, (120, 120, 120), thickness=1, lineType=cv2.LINE_AA)

        mid_torso = ((mid_shoulder[0] + mid_hip[0]) // 2, (mid_shoulder[1] + mid_hip[1]) // 2)
        cv2.ellipse(out, mid_torso, (torso_w, int(torso_w * 0.4)), 0, 0, 360, (100, 100, 100), 1, cv2.LINE_AA)

    # 3. Upper Limbs (Tapered Cylinders)
    # Left Arm
    if vis[5] > conf_thresh and vis[7] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(5), get_pt(7), uarm_r, int(uarm_r * 0.8), c_line, num_rings=3)
    if vis[7] > conf_thresh and vis[9] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(7), get_pt(9), int(uarm_r * 0.8), farm_r, c_line, num_rings=3)

    # Right Arm
    if vis[6] > conf_thresh and vis[8] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(6), get_pt(8), uarm_r, int(uarm_r * 0.8), c_line, num_rings=3)
    if vis[8] > conf_thresh and vis[10] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(8), get_pt(10), int(uarm_r * 0.8), farm_r, c_line, num_rings=3)

    # 4. Lower Limbs (Tapered Cylinders)
    # Left Leg
    if vis[11] > conf_thresh and vis[13] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(11), get_pt(13), thigh_r, int(thigh_r * 0.75), c_line, num_rings=3)
    if vis[13] > conf_thresh and vis[15] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(13), get_pt(15), int(thigh_r * 0.75), calf_r, c_line, num_rings=3)

    # Right Leg
    if vis[12] > conf_thresh and vis[14] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(12), get_pt(14), thigh_r, int(thigh_r * 0.75), c_line, num_rings=3)
    if vis[14] > conf_thresh and vis[16] > conf_thresh:
        draw_cylinder_with_ellipses(out, get_pt(14), get_pt(16), int(thigh_r * 0.75), calf_r, c_line, num_rings=3)

    return out


def create_dual_mode_canvas(
    raw_bgr: np.ndarray,
    coords: np.ndarray,
    volume_params: np.ndarray,
    vis: np.ndarray
) -> np.ndarray:
    """
    Creates a 4-panel master presentation canvas:
    [Original Image] | [Mode 1: Skeleton Wireframe] | [Mode 2: Volumetric 3D Mannequin] | [Combined Artist Canvas]
    """
    h, w, _ = raw_bgr.shape
    blank_white = np.full((h, w, 3), 255, dtype=np.uint8)

    # 1. Overlay Skeleton
    panel_skeleton = draw_skeleton_mode(raw_bgr, coords, vis)

    # 2. Volumetric 3D Mannequin on white sketchbook canvas
    panel_volume = draw_volume_mode(blank_white, coords, volume_params, vis)

    # 3. Combined: Volume primitives overlaid with colored action skeleton
    panel_combined = draw_volume_mode(blank_white, coords, volume_params, vis)
    panel_combined = draw_skeleton_mode(panel_combined, coords, vis)

    # Combine into a horizontal strip
    canvas_4panel = np.hstack([raw_bgr, panel_skeleton, panel_volume, panel_combined])
    return canvas_4panel
