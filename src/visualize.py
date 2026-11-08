"""
Visualizer for Neural Gesture Mannequin:
Mode 1: Skeleton-Based Model (Kinematic Wireframe Rig)
Mode 2: Volume-Based Model (3D Tapered Cylinders, Cross-Contour Ellipses & Torso Box)
"""

import math
from typing import List, Tuple, Optional
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


def generate_joint_driven_anime_sketch(
    coords_np: np.ndarray,
    volume_params: Optional[np.ndarray] = None,
    vis_np: Optional[np.ndarray] = None,
    canvas_size: int = 512,
    style: str = "sepia"  # "sepia", "blue_pencil", or "graphite"
) -> np.ndarray:
    """
    Synthesizes a 2D Anime Mannequin Pose Base (Atari Underdrawing) strictly driven
    by predicted 2D joint coordinates and muscle volume parameters.
    No photorealistic clothes tracing — produces clean structural anatomy guide lines.
    """
    w = h = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Style Color Palettes
    if style == "sepia":
        c_main = (65, 80, 115)     # Warm sepia brown RGB(115, 80, 65)
        c_guide = (120, 135, 160)  # Soft guide lines
        c_joint = (90, 105, 135)   # Pivot joints
    elif style == "blue_pencil":
        c_main = (210, 140, 70)    # Non-photo blue RGB(70, 140, 210)
        c_guide = (230, 180, 130)
        c_joint = (220, 160, 100)
    else:  # graphite
        c_main = (50, 50, 50)
        c_guide = (140, 140, 140)
        c_joint = (90, 90, 90)

    if volume_params is None:
        volume_params = np.array([0.10, 0.18, 0.06, 0.05, 0.09, 0.07], dtype=np.float32)
    if vis_np is None:
        vis_np = np.ones(17, dtype=np.float32)

    # Pixel radii
    head_r = max(18, int(volume_params[0] * w))
    torso_w = max(28, int(volume_params[1] * w))
    uarm_r = max(10, int(volume_params[2] * w))
    farm_r = max(8, int(volume_params[3] * w))
    thigh_r = max(16, int(volume_params[4] * w))
    calf_r = max(12, int(volume_params[5] * w))

    def get_pt(idx):
        return (int(coords_np[idx, 0] * w), int(coords_np[idx, 1] * h))

    # Helper: draw curved anatomical limb with muscle contour
    def draw_muscle_limb(p1, p2, r_top, r_bot):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 8:
            return
        ux, uy = -dy / dist, dx / dist

        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r_mid = max(r_top, r_bot) * 1.12

        pts_left = np.array([
            [int(x1 + ux * r_top), int(y1 + uy * r_top)],
            [int(mid_x + ux * r_mid), int(mid_y + uy * r_mid)],
            [int(x2 + ux * r_bot), int(y2 + uy * r_bot)]
        ], dtype=np.int32)

        pts_right = np.array([
            [int(x1 - ux * r_top), int(y1 - uy * r_top)],
            [int(mid_x - ux * r_mid), int(mid_y - uy * r_mid)],
            [int(x2 - ux * r_bot), int(y2 - uy * r_bot)]
        ], dtype=np.int32)

        cv2.polylines(canvas, [pts_left], isClosed=False, color=c_main, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_right], isClosed=False, color=c_main, thickness=2, lineType=cv2.LINE_AA)

        angle = math.degrees(math.atan2(dy, dx))
        cv2.ellipse(canvas, (int(x2), int(y2)), (r_bot, max(3, int(r_bot * 0.4))), angle, 0, 360, c_guide, 1, cv2.LINE_AA)

    # 1. Torso, Clavicle & Spine
    if vis_np[5] > 0.2 and vis_np[6] > 0.2:
        ls, rs = get_pt(5), get_pt(6)
        lh = get_pt(11) if vis_np[11] > 0.2 else (ls[0], ls[1] + int(h * 0.35))
        rh = get_pt(12) if vis_np[12] > 0.2 else (rs[0], rs[1] + int(h * 0.35))

        mid_sh = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

        # Spinal gesture line
        cv2.line(canvas, mid_sh, mid_hip, c_guide, thickness=1, lineType=cv2.LINE_AA)

        # Clavicle / Collarbone V-line
        neck_base = (mid_sh[0], mid_sh[1] - int(head_r * 0.3))
        cv2.line(canvas, neck_base, ls, c_main, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(canvas, neck_base, rs, c_main, thickness=2, lineType=cv2.LINE_AA)

        # Ribcage Chest Oval
        mid_chest = ((mid_sh[0] * 2 + mid_hip[0]) // 3, (mid_sh[1] * 2 + mid_hip[1]) // 3)
        chest_ry = int(torso_w * 0.9)
        chest_rx = int(torso_w * 0.8)
        cv2.ellipse(canvas, mid_chest, (chest_rx, chest_ry), 0, 0, 360, c_main, thickness=2, lineType=cv2.LINE_AA)

        # Pelvic Hip Bowl
        pelvis_ry = int(torso_w * 0.65)
        pelvis_rx = int(torso_w * 0.75)
        cv2.ellipse(canvas, mid_hip, (pelvis_rx, pelvis_ry), 0, 0, 360, c_main, thickness=2, lineType=cv2.LINE_AA)

        # Torso lateral contour lines
        cv2.line(canvas, (mid_chest[0] - chest_rx, mid_chest[1]), (mid_hip[0] - pelvis_rx, mid_hip[1]), c_main, 2, cv2.LINE_AA)
        cv2.line(canvas, (mid_chest[0] + chest_rx, mid_chest[1]), (mid_hip[0] + pelvis_rx, mid_hip[1]), c_main, 2, cv2.LINE_AA)

    # 2. Head with Facial Crosshairs
    head_center = get_pt(0) if vis_np[0] > 0.2 else ((get_pt(1)[0] + get_pt(2)[0]) // 2, (get_pt(1)[1] + get_pt(2)[1]) // 2)
    cv2.ellipse(canvas, head_center, (head_r, int(head_r * 1.25)), 0, 0, 360, c_main, thickness=2, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, head_center, (head_r, int(head_r * 0.35)), 0, 0, 360, c_guide, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (head_center[0], head_center[1] - int(head_r * 1.25)), (head_center[0], head_center[1] + int(head_r * 1.25)), c_guide, 1, cv2.LINE_AA)

    # 3. Shoulder Pivot Joints
    if vis_np[5] > 0.2:
        cv2.circle(canvas, get_pt(5), int(uarm_r * 0.9), c_joint, 1, cv2.LINE_AA)
    if vis_np[6] > 0.2:
        cv2.circle(canvas, get_pt(6), int(uarm_r * 0.9), c_joint, 1, cv2.LINE_AA)

    # 4. Upper Limbs
    if vis_np[5] > 0.2 and vis_np[7] > 0.2:
        draw_muscle_limb(get_pt(5), get_pt(7), uarm_r, int(uarm_r * 0.8))
    if vis_np[7] > 0.2 and vis_np[9] > 0.2:
        draw_muscle_limb(get_pt(7), get_pt(9), int(uarm_r * 0.8), farm_r)
        cv2.circle(canvas, get_pt(9), int(farm_r * 0.9), c_joint, 1, cv2.LINE_AA)

    if vis_np[6] > 0.2 and vis_np[8] > 0.2:
        draw_muscle_limb(get_pt(6), get_pt(8), uarm_r, int(uarm_r * 0.8))
    if vis_np[8] > 0.2 and vis_np[10] > 0.2:
        draw_muscle_limb(get_pt(8), get_pt(10), int(uarm_r * 0.8), farm_r)
        cv2.circle(canvas, get_pt(10), int(farm_r * 0.9), c_joint, 1, cv2.LINE_AA)

    # 5. Lower Limbs
    if vis_np[11] > 0.2 and vis_np[13] > 0.2:
        draw_muscle_limb(get_pt(11), get_pt(13), thigh_r, int(thigh_r * 0.75))
        cv2.circle(canvas, get_pt(13), int(thigh_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[13] > 0.2 and vis_np[15] > 0.2:
        draw_muscle_limb(get_pt(13), get_pt(15), int(thigh_r * 0.75), calf_r)
        cv2.circle(canvas, get_pt(15), int(calf_r * 0.7), c_joint, 1, cv2.LINE_AA)

    if vis_np[12] > 0.2 and vis_np[14] > 0.2:
        draw_muscle_limb(get_pt(12), get_pt(14), thigh_r, int(thigh_r * 0.75))
        cv2.circle(canvas, get_pt(14), int(thigh_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[14] > 0.2 and vis_np[16] > 0.2:
        draw_muscle_limb(get_pt(14), get_pt(16), int(thigh_r * 0.75), calf_r)
        cv2.circle(canvas, get_pt(16), int(calf_r * 0.7), c_joint, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

