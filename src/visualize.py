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


def get_bezier_curve(p0, p1, p2, p3, num_pts=25):
    t = np.linspace(0, 1, num_pts)[:, None]
    curve = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return curve.astype(np.int32)


def draw_smooth_tapered_curve(canvas, p_start, p_ctrl1, p_ctrl2, p_end, color, base_thickness=2):
    """Draws a smooth cubic Bezier curve with tapering thickness."""
    curve = get_bezier_curve(np.array(p_start), np.array(p_ctrl1), np.array(p_ctrl2), np.array(p_end), num_pts=30)
    for i in range(len(curve) - 1):
        pt1 = tuple(curve[i])
        pt2 = tuple(curve[i + 1])
        progress = i / max(1, len(curve) - 1)
        thick = max(1, int(round(base_thickness * (1.0 + 0.4 * math.sin(progress * math.pi)))))
        cv2.line(canvas, pt1, pt2, color, thickness=thick, lineType=cv2.LINE_AA)


def generate_joint_driven_anime_sketch(
    coords_np: np.ndarray,
    volume_params: Optional[np.ndarray] = None,
    vis_np: Optional[np.ndarray] = None,
    canvas_size: int = 512,
    style: str = "sepia"  # "sepia", "blue_pencil", or "graphite"
) -> np.ndarray:
    """
    Synthesizes an Organic 2D Anime Mannequin Pose Base (Atari Underdrawing) strictly driven
    by predicted 2D joint coordinates and muscle volume parameters.
    Features smooth cubic Bezier curves, tapered line weights, an hourglass torso silhouette,
    an authentic anime jawline, and soft facial crosshair guidelines.
    """
    w = h = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Professional Manga / Anime Color Palettes (BGR for OpenCV)
    if style == "sepia":
        c_core  = (55, 70, 110)    # Deep warm sepia RGB(110, 70, 55)
        c_outer = (75, 95, 135)    # Medium contour brown RGB(135, 95, 75)
        c_guide = (140, 160, 185)  # Soft construction tan RGB(185, 160, 140)
        c_joint = (110, 130, 160)  # Joint pivot ring RGB(160, 130, 110)
    elif style == "blue_pencil":
        c_core  = (200, 120, 50)   # Non-photo blue core RGB(50, 120, 200)
        c_outer = (220, 150, 80)   # Medium blue contour RGB(80, 150, 220)
        c_guide = (240, 195, 140)  # Soft guide blue RGB(140, 195, 240)
        c_joint = (225, 170, 110)  # Pivot cyan RGB(110, 170, 225)
    else:  # graphite pencil
        c_core  = (40, 40, 40)
        c_outer = (70, 70, 70)
        c_guide = (150, 150, 150)
        c_joint = (110, 110, 110)

    if volume_params is None:
        volume_params = np.array([0.10, 0.18, 0.06, 0.05, 0.09, 0.07], dtype=np.float32)
    if vis_np is None:
        vis_np = np.ones(17, dtype=np.float32)

    # Scale radii
    head_r  = max(16, int(volume_params[0] * w))
    torso_w = max(24, int(volume_params[1] * w))
    uarm_r  = max(9, int(volume_params[2] * w))
    farm_r  = max(7, int(volume_params[3] * w))
    thigh_r = max(15, int(volume_params[4] * w))
    calf_r  = max(11, int(volume_params[5] * w))

    def get_pt(idx):
        return (int(round(coords_np[idx, 0] * w)), int(round(coords_np[idx, 1] * h)))

    # 1. Smooth Organic Muscle Limbs with Curved Contours
    def draw_organic_curved_limb(p1, p2, r1, r2, base_thick=2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 8:
            return
        ux, uy = -dy / length, dx / length

        # Outer curve (muscle flare)
        c1_out = (x1 + ux * r1 + dx * 0.3, y1 + uy * r1 + dy * 0.3)
        c2_out = (x1 + ux * (r1 * 1.15) + dx * 0.6, y1 + uy * (r1 * 1.15) + dy * 0.6)
        end_out = (x2 + ux * r2, y2 + uy * r2)

        # Inner curve
        c1_in = (x1 - ux * r1 + dx * 0.3, y1 - uy * r1 + dy * 0.3)
        c2_in = (x1 - ux * (r2 * 1.05) + dx * 0.6, y1 - uy * (r2 * 1.05) + dy * 0.6)
        end_in = (x2 - ux * r2, y2 - uy * r2)

        draw_smooth_tapered_curve(canvas, (x1 + ux * r1, y1 + uy * r1), c1_out, c2_out, end_out, c_outer, base_thick)
        draw_smooth_tapered_curve(canvas, (x1 - ux * r1, y1 - uy * r1), c1_in, c2_in, end_in, c_outer, base_thick)

        # Cross-contour perspective ring at joint
        angle = math.degrees(math.atan2(dy, dx))
        cv2.ellipse(canvas, (int(x2), int(y2)), (r2, max(2, int(r2 * 0.4))), angle, 0, 360, c_guide, 1, cv2.LINE_AA)

    # 2. Torso: Arched Spine (Line of Action), Ribcage Oval & Pelvic Silhouette
    if vis_np[5] > 0.15 and vis_np[6] > 0.15:
        ls, rs = get_pt(5), get_pt(6)
        lh = get_pt(11) if vis_np[11] > 0.15 else (ls[0], ls[1] + int(h * 0.35))
        rh = get_pt(12) if vis_np[12] > 0.15 else (rs[0], rs[1] + int(h * 0.35))

        mid_sh  = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

        # Curving Line of Action (Spine gesture stroke)
        spine_dx, spine_dy = mid_hip[0] - mid_sh[0], mid_hip[1] - mid_sh[1]
        spine_norm = math.hypot(spine_dx, spine_dy)
        if spine_norm > 10:
            sp_ux, sp_uy = -spine_dy / spine_norm, spine_dx / spine_norm
            arch_offset = int(torso_w * 0.25)
            spine_ctrl = (mid_sh[0] + int(spine_dx * 0.5) + int(sp_ux * arch_offset),
                          mid_sh[1] + int(spine_dy * 0.5) + int(sp_uy * arch_offset))
            draw_smooth_tapered_curve(canvas, mid_sh, spine_ctrl, spine_ctrl, mid_hip, c_guide, base_thickness=1)

        # Soft Clavicle V-line
        neck_anchor = (mid_sh[0], mid_sh[1] - int(head_r * 0.35))
        cv2.line(canvas, neck_anchor, ls, c_outer, 2, cv2.LINE_AA)
        cv2.line(canvas, neck_anchor, rs, c_outer, 2, cv2.LINE_AA)
        cv2.circle(canvas, neck_anchor, 3, c_guide, -1, cv2.LINE_AA)

        # Ribcage Egg (tilted along shoulder angle)
        sh_angle = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        mid_chest = (mid_sh[0] + int(spine_dx * 0.35), mid_sh[1] + int(spine_dy * 0.35))
        chest_rx = int(torso_w * 0.85)
        chest_ry = int(torso_w * 0.70)
        cv2.ellipse(canvas, mid_chest, (chest_rx, chest_ry), sh_angle, 0, 360, c_core, 2, cv2.LINE_AA)

        # Pelvis Bowl (tilted along hip angle)
        hip_angle = math.degrees(math.atan2(rh[1] - lh[1], rh[0] - lh[0]))
        pelvis_rx = int(torso_w * 0.78)
        pelvis_ry = int(torso_w * 0.55)
        cv2.ellipse(canvas, mid_hip, (pelvis_rx, pelvis_ry), hip_angle, 0, 360, c_core, 2, cv2.LINE_AA)

        # Flowing hourglass waist lines connecting ribcage to hips
        chest_l = (int(mid_chest[0] - math.cos(math.radians(sh_angle)) * chest_rx),
                   int(mid_chest[1] - math.sin(math.radians(sh_angle)) * chest_rx))
        pelvis_l = (int(mid_hip[0] - math.cos(math.radians(hip_angle)) * pelvis_rx),
                    int(mid_hip[1] - math.sin(math.radians(hip_angle)) * pelvis_rx))
        waist_l_mid = ((chest_l[0] + pelvis_l[0]) // 2 + int(torso_w * 0.15),
                       (chest_l[1] + pelvis_l[1]) // 2)
        draw_smooth_tapered_curve(canvas, chest_l, waist_l_mid, waist_l_mid, pelvis_l, c_core, base_thickness=2)

        chest_r = (int(mid_chest[0] + math.cos(math.radians(sh_angle)) * chest_rx),
                   int(mid_chest[1] + math.sin(math.radians(sh_angle)) * chest_rx))
        pelvis_r = (int(mid_hip[0] + math.cos(math.radians(hip_angle)) * pelvis_rx),
                    int(mid_hip[1] + math.sin(math.radians(hip_angle)) * pelvis_rx))
        waist_r_mid = ((chest_r[0] + pelvis_r[0]) // 2 - int(torso_w * 0.15),
                       (chest_r[1] + pelvis_r[1]) // 2)
        draw_smooth_tapered_curve(canvas, chest_r, waist_r_mid, waist_r_mid, pelvis_r, c_core, base_thickness=2)

    # 3. Head: Tapered Anime Jawline & Facial Crosshairs
    head_pos = get_pt(0) if vis_np[0] > 0.15 else ((get_pt(1)[0] + get_pt(2)[0]) // 2, (get_pt(1)[1] + get_pt(2)[1]) // 2)

    head_tilt = 0.0
    if vis_np[1] > 0.15 and vis_np[2] > 0.15:
        p_le, p_re = get_pt(1), get_pt(2)
        head_tilt = math.degrees(math.atan2(p_re[1] - p_le[1], p_re[0] - p_le[0]))
    elif vis_np[3] > 0.15 and vis_np[4] > 0.15:
        p_la, p_ra = get_pt(3), get_pt(4)
        head_tilt = math.degrees(math.atan2(p_ra[1] - p_la[1], p_ra[0] - p_la[0]))

    # Cranium
    cv2.ellipse(canvas, head_pos, (head_r, int(head_r * 1.15)), head_tilt, 0, 360, c_core, 2, cv2.LINE_AA)

    # Tapered Anime Chin Contour (Soft V-jaw)
    chin_dist = int(head_r * 1.35)
    rad_t = math.radians(head_tilt + 90)
    chin_pt = (int(head_pos[0] + math.cos(rad_t) * chin_dist),
               int(head_pos[1] + math.sin(rad_t) * chin_dist))
    jaw_l = (int(head_pos[0] - math.cos(math.radians(head_tilt)) * head_r * 0.85 + math.cos(rad_t) * head_r * 0.5),
             int(head_pos[1] - math.sin(math.radians(head_tilt)) * head_r * 0.85 + math.sin(rad_t) * head_r * 0.5))
    jaw_r = (int(head_pos[0] + math.cos(math.radians(head_tilt)) * head_r * 0.85 + math.cos(rad_t) * head_r * 0.5),
             int(head_pos[1] + math.sin(math.radians(head_tilt)) * head_r * 0.85 + math.sin(rad_t) * head_r * 0.5))
    draw_smooth_tapered_curve(canvas, jaw_l, jaw_l, chin_pt, chin_pt, c_core, base_thickness=2)
    draw_smooth_tapered_curve(canvas, jaw_r, jaw_r, chin_pt, chin_pt, c_core, base_thickness=2)

    # Facial Crosshairs
    cv2.ellipse(canvas, head_pos, (head_r, int(head_r * 0.32)), head_tilt, 0, 360, c_guide, 1, cv2.LINE_AA)
    top_c = (int(head_pos[0] - math.cos(rad_t) * head_r * 1.1),
             int(head_pos[1] - math.sin(rad_t) * head_r * 1.1))
    cv2.line(canvas, top_c, chin_pt, c_guide, 1, cv2.LINE_AA)

    # 4. Shoulders & Arms
    if vis_np[5] > 0.15: cv2.circle(canvas, get_pt(5), int(uarm_r * 0.9), c_joint, 1, cv2.LINE_AA)
    if vis_np[6] > 0.15: cv2.circle(canvas, get_pt(6), int(uarm_r * 0.9), c_joint, 1, cv2.LINE_AA)

    if vis_np[5] > 0.15 and vis_np[7] > 0.15:
        draw_organic_curved_limb(get_pt(5), get_pt(7), uarm_r, int(uarm_r * 0.82))
        cv2.circle(canvas, get_pt(7), int(uarm_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[7] > 0.15 and vis_np[9] > 0.15:
        draw_organic_curved_limb(get_pt(7), get_pt(9), int(uarm_r * 0.82), farm_r)
        cv2.circle(canvas, get_pt(9), int(farm_r * 0.85), c_joint, 1, cv2.LINE_AA)

    if vis_np[6] > 0.15 and vis_np[8] > 0.15:
        draw_organic_curved_limb(get_pt(6), get_pt(8), uarm_r, int(uarm_r * 0.82))
        cv2.circle(canvas, get_pt(8), int(uarm_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[8] > 0.15 and vis_np[10] > 0.15:
        draw_organic_curved_limb(get_pt(8), get_pt(10), int(uarm_r * 0.82), farm_r)
        cv2.circle(canvas, get_pt(10), int(farm_r * 0.85), c_joint, 1, cv2.LINE_AA)

    # 5. Lower Limbs (Legs)
    if vis_np[11] > 0.15 and vis_np[13] > 0.15:
        draw_organic_curved_limb(get_pt(11), get_pt(13), thigh_r, int(thigh_r * 0.75))
        cv2.circle(canvas, get_pt(13), int(thigh_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[13] > 0.15 and vis_np[15] > 0.15:
        draw_organic_curved_limb(get_pt(13), get_pt(15), int(thigh_r * 0.75), calf_r)
        cv2.circle(canvas, get_pt(15), int(calf_r * 0.7), c_joint, 1, cv2.LINE_AA)

    if vis_np[12] > 0.15 and vis_np[14] > 0.15:
        draw_organic_curved_limb(get_pt(12), get_pt(14), thigh_r, int(thigh_r * 0.75))
        cv2.circle(canvas, get_pt(14), int(thigh_r * 0.7), c_joint, 1, cv2.LINE_AA)
    if vis_np[14] > 0.15 and vis_np[16] > 0.15:
        draw_organic_curved_limb(get_pt(14), get_pt(16), int(thigh_r * 0.75), calf_r)
        cv2.circle(canvas, get_pt(16), int(calf_r * 0.7), c_joint, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


